import os
import logging
from typing import Optional
from dotenv import load_dotenv
from mp_api.client import MPRester

# Load environment variables from the .env file
load_dotenv()

# Securely fetch the key
MP_API_KEY = os.getenv("MATERIALS_PROJECT_API_KEY")

if not MP_API_KEY or MP_API_KEY == "your_actual_api_key_here":
    raise ValueError("Materials Project API key is missing or invalid. Check your .env file.")

def fetch_reference_density(material_id: str) -> Optional[float]:
    """
    Fetches reference density for a given material from the Materials Project.
    Returns the density in g/cm^3, or None if the query fails/material is not found.
    """
    try:
        with MPRester(MP_API_KEY) as mpr:
            # Query the summary endpoint
            doc = mpr.materials.summary.search(material_ids=[material_id])
            
            if doc and len(doc) > 0:
                # The Materials Project returns density in g/cm^3 by default
                return round(doc[0].density, 4)
            else:
                logging.warning(f"Material ID '{material_id}' not found in Materials Project.")
                return None
                
    except Exception as e:
        # Catch network timeouts, API rate limits, or bad requests
        logging.error(f"Failed to fetch data from Materials Project API: {e}")
        return None