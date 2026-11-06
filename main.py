import os
os.environ["FLAGS_cudnn_deterministic"] = "True"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from ollama import AsyncClient

from core.config import (
    CORS_ORIGINS, LLM_CONCURRENCY, LLM_IMAGE_MAX_SIDE, MAX_FILE_SIZE, MAX_PAGES,
    OLLAMA_BASE_URL, OLLAMA_KEEP_ALIVE, OLLAMA_MODEL,
)
from graph.nodes import INVALID_DOC_ERROR, run_extraction_hybrid, aggregate_web_expenses
from graph.ocr_worker import run_ocr, warmup_ocr
from tools.parser import downscale_base64, parse_document_to_images

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("api")

_llm_semaphore = asyncio.Semaphore(LLM_CONCURRENCY)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the OCR model and the Ollama model before accepting requests,
    so the first user doesn't pay the model loading time."""
    try:
        await warmup_ocr()
        logger.info("OCR warm-up done")
    except Exception:
        logger.exception("OCR warm-up failed")
    try:
        # An empty prompt only loads the model into VRAM.
        await AsyncClient(host=OLLAMA_BASE_URL).generate(
            model=OLLAMA_MODEL, prompt="", keep_alive=OLLAMA_KEEP_ALIVE
        )
        logger.info("Ollama model %s loaded", OLLAMA_MODEL)
    except Exception:
        logger.exception("Ollama warm-up failed (%s)", OLLAMA_BASE_URL)
    yield


app = FastAPI(
    title="Financial Report AI",
    description="Upload invoices and receipts to extract expenses using AI",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse_event(step: str, message: str, data: dict | None = None) -> str:
    """Format a Server-Sent Event."""
    payload = {"step": step, "message": message}
    if data:
        payload["data"] = data
    return f"data: {json.dumps(payload)}\n\n"


async def _read_uploads(files: List[UploadFile]) -> list[tuple[str, str, bytes]]:
    """Validate and read every upload once. Returns (filename, content_type, bytes)."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    uploads = []
    for file in files:
        if file.content_type not in SUPPORTED_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type} ({file.filename}). Supported: PDF, PNG, JPG, WEBP"
            )
        file_bytes = await file.read()
        if len(file_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB"
            )
        uploads.append((file.filename, file.content_type, file_bytes))
    return uploads


class PageError(Exception):
    pass


async def _parse_uploads(
    uploads, filenames: list[str], images: list[str], page_files: list[int] | None = None
) -> AsyncIterator[str]:
    """Convert uploads to page images, filling `filenames` and `images`.
    `page_files[i]` (if given) is the index in `filenames` of the file page i came from."""
    yield _sse_event("parsing", "Converting documents to images...")

    for filename, content_type, file_bytes in uploads:
        try:
            pages = await asyncio.to_thread(parse_document_to_images, file_bytes, content_type)
        except Exception as e:
            raise PageError(f"Failed to process '{filename}': {e}")
        images.extend(pages)
        if page_files is not None:
            page_files.extend([len(filenames)] * len(pages))
        filenames.append(filename)

    if not images:
        raise PageError("Could not extract any images from the uploaded documents.")
    if len(images) > MAX_PAGES:
        raise PageError(f"Too many pages ({len(images)}). Maximum is {MAX_PAGES} per request.")

    yield _sse_event("parsed", f"Found {len(images)} pages to process.", {"total_docs": len(images)})


async def _process_pages(
    images: list[str],
    results: list,
    on_page_done: Callable[[int, object], str | None] | None = None,
) -> AsyncIterator[str]:
    """OCR + LLM extraction for all pages, pipelined: while one page is in the LLM,
    the next one is already in OCR. OCR is serialized (one model instance) and LLM
    calls are capped by LLM_CONCURRENCY across all requests.

    Yields SSE progress events; fills `results[i]` with the extraction for page i.
    If `on_page_done(i, result)` returns an event, it is yielded as soon as page i
    is extracted, so the client can show each page's data without waiting for the rest.
    """
    total = len(images)
    results.extend([None] * total)
    events: asyncio.Queue = asyncio.Queue()
    done = object()

    async def handle_page(i: int, img_b64: str) -> None:
        progress = {"current": i + 1, "total": total}

        async def on_ocr_start():
            await events.put(_sse_event("ocr", f"Running OCR text recognition for page {i+1}/{total}...", progress))

        try:
            text = await run_ocr(img_b64, i, total, on_start=on_ocr_start)
        except Exception as e:
            raise PageError(f"OCR failed on page {i+1}: {e}") from e
        logger.debug("OCR page %d/%d text:\n%s", i + 1, total, text)

        llm_image = await asyncio.to_thread(downscale_base64, img_b64, LLM_IMAGE_MAX_SIDE)
        async with _llm_semaphore:
            await events.put(_sse_event("extracting", f"Extracting data with AI for page {i+1}/{total}...", progress))
            try:
                results[i] = await run_extraction_hybrid(text, llm_image, i)
            except Exception as e:
                raise PageError(f"AI extraction failed on page {i+1}: {e}") from e
        if on_page_done:
            event = on_page_done(i, results[i])
            if event:
                await events.put(event)

    async def run(i: int, img_b64: str) -> None:
        try:
            await handle_page(i, img_b64)
            await events.put(done)
        except Exception as e:
            await events.put(e)

    tasks = [asyncio.create_task(run(i, img)) for i, img in enumerate(images)]
    try:
        remaining = total
        while remaining:
            item = await events.get()
            if item is done:
                remaining -= 1
            elif isinstance(item, Exception):
                raise item
            else:
                yield item
    finally:
        # Client disconnected or a page failed: stop the pages still waiting.
        for task in tasks:
            task.cancel()


def _flatten(results: list) -> list:
    expenses = []
    for result in results:
        if result == INVALID_DOC_ERROR:
            expenses.append(INVALID_DOC_ERROR)
        elif isinstance(result, list):
            expenses.extend(result)
    return expenses


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "financial-report-ai"}


@app.post("/api/report-stream")
async def generate_financial_report_stream(
    files: List[UploadFile] = File(...)
):
    """Upload one or more financial documents and stream progress via SSE (Mobile flow)."""
    uploads = await _read_uploads(files)

    async def event_generator():
        filenames: list[str] = []
        images: list[str] = []
        results: list = []
        try:
            async for event in _parse_uploads(uploads, filenames, images):
                yield event
            async for event in _process_pages(images, results):
                yield event

            yield _sse_event("formatting", "Formatting report...")
            extracted_expenses = _flatten(results)
            total_docs = len(images)

            if extracted_expenses and extracted_expenses[0] == INVALID_DOC_ERROR:
                yield _sse_event("done", "Extraction complete!", {
                    "success": False,
                    "error": "The uploaded document does not appear to be a fiscal receipt or invoice. Please upload a valid receipt.",
                    "pages_processed": total_docs,
                    "files": filenames,
                    "expenses": {},
                })
                return

            expenses_obj = {}
            if extracted_expenses:
                # API mobile just takes the first expense
                exp = extracted_expenses[0]
                invoice_str = exp.invoice_number_date
                if exp.receipt_date and exp.receipt_date not in invoice_str:
                    invoice_str = f"{invoice_str} / {exp.receipt_date}"

                expenses_obj = {
                    "expense_description": exp.expense_description,
                    "invoice_number_date": invoice_str,
                    "expense_amount": exp.expense_amount,
                    "currency": exp.currency,
                    "expense_category": exp.expense_category,
                }

            yield _sse_event("done", "Extraction complete!", {
                "success": True,
                "report": "### Business Trip Expenses Extracted Successfully.",
                "pages_processed": total_docs,
                "files": filenames,
                "expenses": expenses_obj,
            })

        except PageError as e:
            yield _sse_event("error", str(e))
        except Exception as e:
            logger.exception("report-stream failed")
            yield _sse_event("error", f"Unexpected error: {str(e)}")

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/api/web-report-stream")
async def generate_web_report_stream(
    files: List[UploadFile] = File(...)
):
    """Process multiple documents and stream progress via SSE (Web flow)."""
    uploads = await _read_uploads(files)

    async def event_generator():
        filenames: list[str] = []
        images: list[str] = []
        page_files: list[int] = []
        results: list = []

        def page_item_event(i: int, result) -> str:
            """Send each page's expenses as soon as they are extracted. The final
            `done` event still carries all items, for clients that ignore `item`."""
            items = aggregate_web_expenses(_flatten([result]))["web_result"].get("items", [])
            file_index = page_files[i]
            return _sse_event("item", f"Page {i+1}/{len(images)} extracted.", {
                "page": i + 1,
                "total": len(images),
                "file_index": file_index,
                "file": filenames[file_index],
                "valid": result != INVALID_DOC_ERROR,
                "items": items,
            })

        try:
            async for event in _parse_uploads(uploads, filenames, images, page_files):
                yield event
            async for event in _process_pages(images, results, page_item_event):
                yield event

            yield _sse_event("aggregating", "Aggregating extracted expenses...")
            web_result = aggregate_web_expenses(_flatten(results))["web_result"]

            if not web_result.get("items"):
                yield _sse_event("error", "The AI could not extract any expenses from the uploaded documents.")
                return

            yield _sse_event("done", "Extraction complete!", {
                "pages_processed": len(images),
                "files": filenames,
                **web_result
            })

        except PageError as e:
            yield _sse_event("error", str(e))
        except Exception as e:
            logger.exception("web-report-stream failed")
            yield _sse_event("error", f"Unexpected error: {str(e)}")

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=SSE_HEADERS)
