import os
from dotenv import load_dotenv
from mp_api.client import MPRester

# Load environment variables from the .env file
load_dotenv()

# Securely fetch the key
MP_API_KEY = os.getenv("MATERIALS_PROJECT_API_KEY")

if not MP_API_KEY:
    raise ValueError("Materials Project API key is missing. Check your .env file.")

def fetch_reference_density(material_id: str) -> float:
    """Fetches reference density for a given material from the Materials Project."""
    with MPRester(MP_API_KEY) as mpr:
        # Query the summary endpoint
        doc = mpr.materials.summary.search(material_ids=[material_id])
        if doc and len(doc) > 0:
            return doc[0].density
        return None