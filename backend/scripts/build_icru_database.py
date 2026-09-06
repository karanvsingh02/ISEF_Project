import json
import os
from mendeleev import element as mendeleev_element

# Official ICRU Report 37 / NIST PSTAR Mean Excitation Energies (I) in eV
# Source: https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html
ICRU_I_VALUES_EV = {
    1: 19.2, 2: 41.8, 3: 40.0, 4: 63.7, 5: 76.0, 6: 78.0, 7: 82.0, 8: 95.0, 9: 115.0, 10: 137.0,
    11: 149.0, 12: 156.0, 13: 166.0, 14: 173.0, 15: 173.0, 16: 180.0, 17: 174.0, 18: 188.0, 19: 190.0, 20: 191.0,
    21: 216.0, 22: 233.0, 23: 243.0, 24: 257.0, 25: 272.0, 26: 286.0, 27: 297.0, 28: 311.0, 29: 322.0, 30: 330.0,
    31: 334.0, 32: 350.0, 33: 347.0, 34: 348.0, 35: 343.0, 36: 352.0, 37: 363.0, 38: 366.0, 39: 379.0, 40: 393.0,
    41: 417.0, 42: 424.0, 43: 428.0, 44: 441.0, 45: 449.0, 46: 470.0, 47: 470.0, 48: 469.0, 49: 488.0, 50: 488.0,
    51: 487.0, 52: 485.0, 53: 491.0, 54: 482.0, 55: 488.0, 56: 491.0, 57: 501.0, 58: 523.0, 59: 535.0, 60: 546.0,
    61: 560.0, 62: 574.0, 63: 580.0, 64: 591.0, 65: 614.0, 66: 628.0, 67: 650.0, 68: 658.0, 69: 674.0, 70: 684.0,
    71: 694.0, 72: 705.0, 73: 718.0, 74: 727.0, 75: 736.0, 76: 746.0, 77: 757.0, 78: 790.0, 79: 790.0, 80: 800.0,
    81: 810.0, 82: 823.0, 83: 823.0, 84: 830.0, 85: 825.0, 86: 794.0, 87: 827.0, 88: 826.0, 89: 841.0, 90: 847.0,
    91: 878.0, 92: 890.0
}

def build_icru_database():
    print("Building ICRU-37 Materials Database...")
    database = {}

    for z in range(1, 93):
        el = mendeleev_element(z)
        symbol = el.symbol
        
        # Structure the data exactly how our physics engine needs it
        database[symbol] = {
            "Z": z,
            "A": round(el.atomic_weight, 4),
            "I_eV": ICRU_I_VALUES_EV[z]
        }
        print(f"Loaded {symbol:<2} (Z={z:<2}) | A={database[symbol]['A']:<8} | I={database[symbol]['I_eV']} eV")

    # Save to the app/data folder
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "app", "data", "icru37_data.json"
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(database, f, indent=2)

    print(f"\n✅ Success! Wrote 92 elements to {output_path}")

if __name__ == "__main__":
    build_icru_database()