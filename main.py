from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List
import traceback
import json
import asyncio
import os
os.environ["FLAGS_cudnn_deterministic"] = "True"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"
import requests
from jose import jwt, JWTError
from graph.builder import create_graph
from graph.nodes import run_ocr_single, _run_extraction_hybrid, aggregate_web_expenses
from tools.parser import parse_document_to_images

app = FastAPI(
    title="Financial Report AI",
    description="Upload bank statements, invoices, and receipts to generate financial reports using AI",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

graph = create_graph()

SUPPORTED_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB per file

security = HTTPBearer()

def _sse_event(step: str, message: str, data: dict | None = None) -> str:
    """Format a Server-Sent Event."""
    payload = {"step": step, "message": message}
    if data:
        payload["data"] = data
    return f"data: {json.dumps(payload)}\n\n"

KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:8080/realms/master")

CLIENT_ID = os.getenv("CLIENT_ID", "fastapi-image-to-text")

try:
    jwks_url = f"{KEYCLOAK_URL}/protocol/openid-connect/certs"
    jwks = requests.get(jwks_url).json()
    print("[Auth] Cheile publice Keycloak au fost încărcate cu succes.")
except Exception as e:
    print(f"[Auth] Eroare la obținerea JWKS de la Keycloak: {e}")
    jwks = {"keys": []}

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Dependință care validează token-ul JWT la fiecare apel.
    """
    token = credentials.credentials
    try:
        unverified_header = jwt.get_unverified_header(token)
        rsa_key = {}
        for key in jwks.get("keys", []):
            if key["kid"] == unverified_header.get("kid"):
                rsa_key = {
                    "kty": key["kty"],
                    "kid": key["kid"],
                    "use": key["use"],
                    "n": key["n"],
                    "e": key["e"]
                }
                break
                
        if not rsa_key:
            raise HTTPException(status_code=401, detail="Cheia publică pentru token nu a fost găsită.")

        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=[unverified_header["alg"]],
            issuer=KEYCLOAK_URL,
            options={"verify_aud": False}
        )
        return payload

    except JWTError as e:
        raise HTTPException(status_code=401, detail=f"Token invalid sau expirat: {str(e)}")



@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "financial-report-ai"}


@app.get("/api/secure-health")
# async def secure_health_check(user_payload: dict = Depends(verify_token)):
async def secure_health_check():
    """
    Endpoint simplu pentru a testa rapid autentificarea Keycloak.
    """
    return {
        "status": "success",
        "message": "Autentificarea a funcționat perfect! (Validare dezactivata)",
        "client_conectat": "FaraAuth" # user_payload.get("azp", "Necunoscut")
    }



@app.get("/api/graph")
# async def get_graph_image(user_payload: dict = Depends(verify_token)):
async def get_graph_image():
    """Returns the LangGraph architecture as a PNG image."""
    try:
        img_bytes = graph.get_graph().draw_mermaid_png()
        return Response(content=img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate graph image: {str(e)}"
        )


@app.post("/api/report")
async def generate_financial_report(
    files: List[UploadFile] = File(...)
    # user_payload: dict = Depends(verify_token)
):
    """Upload one or more financial documents and generate a report."""

    # print(f"[API] Apel efectuat de clientul: {user_payload.get('azp', 'Necunoscut')}")
    print("[API] Apel efectuat (validare token dezactivata)")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    all_document_images: list[str] = []
    filenames: list[str] = []

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

        try:
            images = parse_document_to_images(file_bytes, file.content_type)
            all_document_images.extend(images)
            filenames.append(file.filename)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to process '{file.filename}': {str(e)}"
            )

    if not all_document_images:
        raise HTTPException(
            status_code=422,
            detail="Could not extract any images from the uploaded documents."
        )

    print(f"[API] Processing {len(files)} file(s), {len(all_document_images)} page(s) total")

    try:
        result = graph.invoke({
            "documents": all_document_images,
            "source": "mobile",
            "extracted_texts": [],
            "extracted_expenses": [],
            "current_doc_index": 0,
            "report": "",
            "web_result": {},
            "company_name": "Nexus Digital",
            "company_cif": "RO38492011",
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Report generation failed: {str(e)}"
        )

    extracted = result.get("extracted_expenses", [])
    
    if extracted and extracted[0] == "__INVALID_DOCUMENT__":
        return {
            "success": False,
            "error": "The uploaded document does not appear to be a fiscal receipt or invoice. Please upload a valid receipt.",
            "pages_processed": len(all_document_images),
            "files": filenames,
            "expenses": {},
        }
    
    if extracted:
        exp = extracted[0]
        invoice_str = exp.invoice_number_date
        if exp.receipt_date:
            invoice_str = f"{invoice_str} / {exp.receipt_date}"
        
        expenses_obj = {
            "expense_description": exp.expense_description,
            "invoice_number_date": invoice_str,
            "expense_amount": exp.expense_amount,
            "currency": exp.currency,
        }
    else:
        expenses_obj = {}

    return {
        "success": True,
        "report": result.get("report", ""),
        "pages_processed": len(all_document_images),
        "files": filenames,
        "extracted_texts": result.get("extracted_texts", []),
        "expenses": expenses_obj,
    }


@app.post("/api/report-stream")
async def generate_financial_report_stream(
    files: List[UploadFile] = File(...)
):
    """Upload one or more financial documents and stream progress via SSE (Mobile flow)."""

    print("[API] Apel report-stream efectuat")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

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
        await file.seek(0) # reset file pointer

    async def event_generator():
        try:
            yield _sse_event("parsing", "Converting documents to images...")
            
            all_document_images = []
            filenames = []
            
            for file in files:
                file_bytes = await file.read()
                try:
                    images = await asyncio.to_thread(parse_document_to_images, file_bytes, file.content_type)
                    all_document_images.extend(images)
                    filenames.append(file.filename)
                except Exception as e:
                    yield _sse_event("error", f"Failed to process '{file.filename}': {str(e)}")
                    return

            if not all_document_images:
                yield _sse_event("error", "Could not extract any images from the uploaded documents.")
                return

            total_docs = len(all_document_images)
            yield _sse_event("parsed", f"Found {total_docs} pages to process.", {"total_docs": total_docs})

            extracted_expenses = []

            for i, doc_b64 in enumerate(all_document_images):
                yield _sse_event("ocr", f"Running OCR text recognition for page {i+1}/{total_docs}...", {"current": i+1, "total": total_docs})
                try:
                    extracted_text = await asyncio.to_thread(run_ocr_single, doc_b64, i, total_docs)
                except Exception as e:
                    yield _sse_event("error", f"OCR failed on page {i+1}: {str(e)}")
                    return

                print(f"\n{'='*60}\n[OCR-FULL-TEXT] doc_index={i+1}/{total_docs}\n{extracted_text}\n{'='*60}\n")

                yield _sse_event("extracting", f"Extracting data with AI for page {i+1}/{total_docs}...", {"current": i+1, "total": total_docs})
                try:
                    result = await asyncio.to_thread(_run_extraction_hybrid, extracted_text, doc_b64, i)
                except Exception as e:
                    yield _sse_event("error", f"AI extraction failed on page {i+1}: {str(e)}")
                    return

                if result == "__INVALID_DOCUMENT__":
                    extracted_expenses.append("__INVALID_DOCUMENT__")
                elif isinstance(result, list):
                    extracted_expenses.extend(result)

            yield _sse_event("formatting", "Formatting report...")
            
            # Replicate the mobile report formatting logic
            if extracted_expenses and extracted_expenses[0] == "__INVALID_DOCUMENT__":
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
                if exp.receipt_date:
                    invoice_str = f"{invoice_str} / {exp.receipt_date}"
                
                expenses_obj = {
                    "expense_description": exp.expense_description,
                    "invoice_number_date": invoice_str,
                    "expense_amount": exp.expense_amount,
                    "currency": exp.currency,
                }
            
            yield _sse_event("done", "Extraction complete!", {
                "success": True,
                "report": "### Business Trip Expenses Extracted Successfully.",
                "pages_processed": total_docs,
                "files": filenames,
                "expenses": expenses_obj,
            })

        except Exception as e:
            traceback.print_exc()
            yield _sse_event("error", f"Unexpected error: {str(e)}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post("/api/web-report")
async def generate_web_report(
    files: List[UploadFile] = File(...)
):
    """Process multiple documents and return categorized expenses for web form auto-fill."""

    print("[API-WEB] Apel web-report")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    all_document_images: list[str] = []
    filenames: list[str] = []

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

        try:
            images = parse_document_to_images(file_bytes, file.content_type)
            all_document_images.extend(images)
            filenames.append(file.filename)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Failed to process '{file.filename}': {str(e)}"
            )

    if not all_document_images:
        raise HTTPException(
            status_code=422,
            detail="Could not extract any images from the uploaded documents."
        )

    print(f"[API-WEB] Processing {len(files)} file(s), {len(all_document_images)} page(s) total")

    try:
        result = graph.invoke({
            "documents": all_document_images,
            "source": "web",
            "extracted_texts": [],
            "extracted_expenses": [],
            "current_doc_index": 0,
            "report": "",
            "web_result": {},
            "company_name": "Nexus Digital",
            "company_cif": "RO38492011",
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Web report generation failed: {str(e)}"
        )

    web_result = result.get("web_result", {})
    items = web_result.get("items", [])

    if not items:
        return {
            "success": False,
            "error": "The AI could not extract any expenses from the uploaded documents.",
            "pages_processed": len(all_document_images),
            "files": filenames,
        }

    return {
        "success": True,
        "pages_processed": len(all_document_images),
        "files": filenames,
        **web_result,
    }


@app.post("/api/web-report-stream")
async def generate_web_report_stream(
    files: List[UploadFile] = File(...)
):
    """Process multiple documents and stream progress via SSE."""

    print("[API-WEB] Apel web-report-stream")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

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
        await file.seek(0) # reset file pointer

    async def event_generator():
        try:
            yield _sse_event("parsing", "Converting documents to images...")
            
            all_document_images = []
            filenames = []
            
            for file in files:
                file_bytes = await file.read()
                try:
                    images = await asyncio.to_thread(parse_document_to_images, file_bytes, file.content_type)
                    all_document_images.extend(images)
                    filenames.append(file.filename)
                except Exception as e:
                    yield _sse_event("error", f"Failed to process '{file.filename}': {str(e)}")
                    return

            if not all_document_images:
                yield _sse_event("error", "Could not extract any images from the uploaded documents.")
                return

            total_docs = len(all_document_images)
            yield _sse_event("parsed", f"Found {total_docs} pages to process.", {"total_docs": total_docs})

            extracted_expenses = []

            for i, doc_b64 in enumerate(all_document_images):
                yield _sse_event("ocr", f"Running OCR text recognition for page {i+1}/{total_docs}...", {"current": i+1, "total": total_docs})
                try:
                    extracted_text = await asyncio.to_thread(run_ocr_single, doc_b64, i, total_docs)
                except Exception as e:
                    yield _sse_event("error", f"OCR failed on page {i+1}: {str(e)}")
                    return

                print(f"\n{'='*60}\n[OCR-FULL-TEXT] doc_index={i+1}/{total_docs}\n{extracted_text}\n{'='*60}\n")

                yield _sse_event("extracting", f"Extracting data with AI for page {i+1}/{total_docs}...", {"current": i+1, "total": total_docs})
                try:
                    result = await asyncio.to_thread(_run_extraction_hybrid, extracted_text, doc_b64, i)
                except Exception as e:
                    yield _sse_event("error", f"AI extraction failed on page {i+1}: {str(e)}")
                    return

                if result == "__INVALID_DOCUMENT__":
                    extracted_expenses.append("__INVALID_DOCUMENT__")
                elif isinstance(result, list):
                    extracted_expenses.extend(result)

            yield _sse_event("aggregating", "Aggregating extracted expenses...")
            
            # Use existing logic to group expenses by type and calculate totals
            state_mock = {"extracted_expenses": extracted_expenses}
            aggregated = aggregate_web_expenses(state_mock)
            web_result = aggregated.get("web_result", {})

            if not web_result.get("items"):
                yield _sse_event("error", "The AI could not extract any expenses from the uploaded documents.")
                return

            yield _sse_event("done", "Extraction complete!", {
                "pages_processed": total_docs,
                "files": filenames,
                **web_result
            })

        except Exception as e:
            traceback.print_exc()
            yield _sse_event("error", f"Unexpected error: {str(e)}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )