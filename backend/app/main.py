from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import predict
from app.routers import materials

app = FastAPI(
    title="Space Radiation Shielding Closed-Loop System",
    description="Backend service for generative design, PINN surrogate modeling, and simulation orchestration.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # <-- FIX: Prevents the Starlette startup crash
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include prediction router
app.include_router(predict.router)
app.include_router(materials.router)

@app.get("/")
async def root():
    return {"status": "active", "message": "Radiation Shielding Engine API is running."}