from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

from app.core.rdkit_service import parse_smiles_composition, KNOWN_POLYMERS
from app.core.bicerano import calculate_bicerano_properties

router = APIRouter(prefix="/api/v1/materials", tags=["Materials & Cheminformatics"])

class MaterialAnalysisRequest(BaseModel):
    smiles: str

class MaterialAnalysisResponse(BaseModel):
    smiles: str
    hydrogen_mass_fraction: float
    carbon_mass_fraction: float
    oxygen_mass_fraction: float
    density_g_cm3: float
    mean_excitation_energy_ev: float
    status: str

@router.post("/analyze-smiles", response_model=MaterialAnalysisResponse)
async def analyze_smiles(payload: MaterialAnalysisRequest):
    try:
        comp = parse_smiles_composition(payload.smiles)
        props = calculate_bicerano_properties(
            comp["hydrogen_fraction_wH"],
            comp["carbon_fraction_wC"],
            comp["oxygen_fraction_wO"]
        )
        return MaterialAnalysisResponse(
            smiles=payload.smiles,
            hydrogen_mass_fraction=props["hydrogen_mass_fraction"],
            carbon_mass_fraction=round(comp["carbon_fraction_wC"], 4),
            oxygen_mass_fraction=round(comp["oxygen_fraction_wO"], 4),
            density_g_cm3=props["density_g_cm3"],
            mean_excitation_energy_ev=props["mean_excitation_energy_ev"],
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process SMILES string: {str(e)}")

@router.get("/known-polymers")
async def get_known_polymers() -> Dict[str, str]:
    return KNOWN_POLYMERS