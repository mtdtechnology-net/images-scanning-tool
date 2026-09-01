from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import traceback
import os
import requests
from jose import jwt, JWTError
from graph.builder import create_graph
from tools.parser import parse_document_to_images

app = FastAPI(
    title="Financial Report AI",
    description="Upload bank statements, invoices, and receipts to generate financial reports using AI",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
async def secure_health_check(user_payload: dict = Depends(verify_token)):
    """
    Endpoint simplu pentru a testa rapid autentificarea Keycloak.
    """
    return {
        "status": "success",
        "message": "Autentificarea a funcționat perfect!",
        "client_conectat": user_payload.get("azp", "Necunoscut")
    }



@app.get("/api/graph")
async def get_graph_image(user_payload: dict = Depends(verify_token)):
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
    files: List[UploadFile] = File(...),
    user_payload: dict = Depends(verify_token)
):
    """Upload one or more financial documents and generate a report."""

    print(f"[API] Apel efectuat de clientul: {user_payload.get('azp', 'Necunoscut')}")

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
            "extracted_texts": [],
            "extracted_expenses": [],
            "current_doc_index": 0,
            "report": "",
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