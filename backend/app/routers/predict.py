import os
import torch
from fastapi import APIRouter
from pydantic import BaseModel
from app.models.pinn import ShieldingPINN

router = APIRouter(prefix="/api/v1/predict", tags=["PINN Prediction"])

# Initialize PINN model instance and load trained weights
model = ShieldingPINN()
weight_path = os.path.join("app", "weights", "pinn_v1.pt")

if os.path.exists(weight_path):
    model.load_state_dict(torch.load(weight_path, weights_only=True))
    print(f"✅ Loaded trained PINN weights from {weight_path}")
else:
    print("⚠️ Warning: pinn_v1.pt not found. Using untrained weights.")

model.eval()

class PredictionRequest(BaseModel):
    thickness_cm: float
    energy_mev: float

class PredictionResponse(BaseModel):
    thickness_cm: float
    energy_mev: float
    predicted_dose_mgy: float
    status: str

@router.post("/pinn", response_model=PredictionResponse)
async def predict_dose(payload: PredictionRequest):
    input_tensor = torch.tensor([[payload.thickness_cm, payload.energy_mev]], dtype=torch.float32)
    
    with torch.no_grad():
        predicted_dose = model(input_tensor).item()
        
    return PredictionResponse(
        thickness_cm=payload.thickness_cm,
        energy_mev=payload.energy_mev,
        predicted_dose_mgy=max(0.0, predicted_dose),
        status="success"
    )