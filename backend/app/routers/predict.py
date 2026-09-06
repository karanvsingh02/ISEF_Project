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
    w_regolith: float
    density_g_cm3: float
    topology_idx: float

class PredictionResponse(BaseModel):
    thickness_cm: float
    w_regolith: float
    predicted_dose_mgy: float
    predicted_neutron_flux: float
    status: str

@router.post("/pinn", response_model=PredictionResponse)
async def predict_dose(payload: PredictionRequest):
    # 1. Align tensor exactly with PINN input layer [Thickness, w_regolith, Density, Topology_Index]
    input_tensor = torch.tensor([[
        payload.thickness_cm, 
        payload.w_regolith, 
        payload.density_g_cm3, 
        payload.topology_idx
    ]], dtype=torch.float32)
    
    with torch.no_grad():
        predictions = model(input_tensor)
        
        # 2. Extract both Dose (index 0) and Neutron Flux (index 1)
        predicted_dose = predictions[0, 0].item()
        predicted_flux = predictions[0, 1].item()
        
    return PredictionResponse(
        thickness_cm=payload.thickness_cm,
        w_regolith=payload.w_regolith,
        predicted_dose_mgy=max(0.0, predicted_dose),     # ReLU-style safety clamp
        predicted_neutron_flux=max(0.0, predicted_flux), # ReLU-style safety clamp
        status="success"
    )