import os
import sys
import torch

# Add backend directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.models.pinn import ShieldingPINN

def validate():
    
    print("🔬 Initializing NIST PSTAR Ground-Truth Validation...")
    
    model = ShieldingPINN()
    weight_path = os.path.join("app", "weights", "pinn_v1.pt")
    
    if not os.path.exists(weight_path):
        print(f"❌ Error: {weight_path} not found. Train the model first using train_pinn.py.")
        return

    model.load_state_dict(torch.load(weight_path, weights_only=True))
    model.eval()

    # Reference values based on NIST PSTAR proton CSDA ranges in Polyethylene
    # 50 MeV  --> CSDA Range ~ 2.4 cm
    # 100 MeV --> CSDA Range ~ 8.2 cm
    # 200 MeV --> CSDA Range ~ 28.6 cm
    test_cases = [
        {"energy": 50.0, "thickness": 1.0, "note": "Below CSDA range (2.4 cm) - Should transmit dose"},
        {"energy": 50.0, "thickness": 3.0, "note": "Beyond CSDA range (2.4 cm) - Should drop to ~0 mGy"},
        {"energy": 100.0, "thickness": 5.0, "note": "Below CSDA range (8.2 cm) - Should transmit dose"},
        {"energy": 100.0, "thickness": 10.0, "note": "Beyond CSDA range (8.2 cm) - Should drop to ~0 mGy"},
        {"energy": 200.0, "thickness": 15.0, "note": "Below CSDA range (28.6 cm) - Should transmit dose"}
    ]

    print("\n" + "="*80)
    print(f"{'Energy (MeV)':<14} | {'Thickness (cm)':<16} | {'Predicted Dose (mGy)':<22} | {'Physics Expectation'}")
    print("="*80)

    with torch.no_grad():
        for case in test_cases:
            x_in = torch.tensor([[case["thickness"], case["energy"]]], dtype=torch.float32)
            dose = max(0.0, model(x_in).item())
            print(f"{case['energy']:<14.1f} | {case['thickness']:<16.1f} | {dose:<22.4f} | {case['note']}")

    print("="*80 + "\n")

if __name__ == "__main__":
    validate()