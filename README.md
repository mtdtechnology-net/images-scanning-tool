# Financial Report AI - Backend

This is the backend service for the Financial Report AI application. It uses **FastAPI**, **LangGraph**, **PaddleOCR**, and **Ollama** (LLM) to process financial documents (PDFs, images) and extract structured data using AI.

---

## 🏗️ Architecture & DevOps Notes (For Miron)

Please read this section carefully before deploying, as this application has specific hardware and networking requirements:

### 1. Hardware Bottlenecks & Requirements
- **CPU (Heavy Usage):** The backend relies heavily on `PaddleOCR` to extract text from images *before* sending data to the LLM. To speed this up, it uses a `ProcessPoolExecutor` which spawns multiple isolated processes concurrently. **This is a massive CPU hog.** The backend Docker container/server needs generous CPU limits (minimum 4-8 vCPUs) to prevent throttling or crashes during concurrent user requests.
- **RAM:** Generous RAM (4-8GB+) is recommended to hold large PDFs, base64 images, and the OCR models in memory.
- **GPU (Ollama):** The backend communicates with an Ollama server (currently pointing to `http://192.168.100.56:11434`). The Ollama server absolutely needs a dedicated GPU (e.g., NVIDIA) and sufficient VRAM to handle multiple contexts (if `OLLAMA_NUM_PARALLEL` is configured on the Ollama side).

### 2. Statelessness & Kubernetes
- The backend is **100% Stateless**. 
- It does not save files to disk (everything is processed in memory) and it does not use a database (no SQL, Redis, etc.).
- It is perfectly safe to deploy in a Kubernetes cluster or Docker Swarm. You do not need to configure any persistent volume claims (PVCs).

### 3. Authentication (Keycloak)
The backend expects an `Authorization: Bearer <token>` header for its protected endpoints. 
Authentication is implemented via Keycloak JWT verification.
- Ensure the `KEYCLOAK_URL` and `CLIENT_ID` environment variables are properly set in the deployment environment.

---

## ⚙️ Environment Variables

The application can be configured using the following environment variables (set these in your Dockerfile, `docker-compose.yml`, or Kubernetes ConfigMap):

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `KEYCLOAK_URL` | `http://localhost:8080/realms/master` | The URL of the Keycloak realm used to fetch the JWKS public keys for token validation. |
| `CLIENT_ID` | `fastapi-image-to-text` | The expected audience / client ID for the JWT token. |

*(Note: The Ollama server URL is currently configured in `graph/nodes.py`. Ensure the backend container can route traffic to `http://192.168.100.56:11434` or update the code to use an environment variable for the LLM host).*

---

## 🚀 How to Run Locally

The project uses modern Python (`>=3.12`) and dependencies can be managed via `uv` or `pip`.

### 1. Install Dependencies
If using `uv` (recommended for speed):
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```
*(Alternatively, use standard pip: `pip install -r pyproject.toml` equivalent)*

### 2. System Dependencies for OCR (Linux/Docker)
If you are building a Docker image, `PaddleOCR` requires OpenCV system dependencies. Ensure your `Dockerfile` includes:
```dockerfile
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0
```

### 3. Start the Server
Run the FastAPI application using Uvicorn:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```
Or simply:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`
Swagger UI Documentation: `http://localhost:8000/docs`

---

## 🔍 Health Checks
For Load Balancers or Kubernetes readiness/liveness probes, a public health check endpoint is available at:
- **`GET /api/health`** (Returns `{"status": "ok", ...}`)
