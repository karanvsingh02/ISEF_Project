from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(
    title="Space Radiation Shielding Closed-Loop System",
    description="Backend service for generative design, PINN surrogate modeling, and simulation orchestration.",
    version="1.0.0",
)

# Configure CORS headers for local API communication
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "*"  # Allows desktop client connections during development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthCheckResponse(BaseModel):
    status: str
    message: str


@app.get("/", response_model=HealthCheckResponse)
async def root():
    return HealthCheckResponse(
        status="active",
        message="Radiation Shielding AI/Sim Engine API is running."
    )


@app.get("/health")
async def health_check():
    return {"status": "ok"}