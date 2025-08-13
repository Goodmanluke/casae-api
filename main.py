from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Casae API", version="0.1.0")

# CORS configuration will be overridden via env vars (ALLOWED_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # to be replaced with config
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Welcome to Casae API"}