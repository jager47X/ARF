# convert_cfr_json.py
"""
Convert CFR JSON from old format (titles) to new format (regulations)
to match the structure of other JSON files like ca_regulations.json
"""
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("convert_cfr_json")

def convert_cfr_json(input_path: Path, output_path: Path = None):
    """Convert CFR JSON from 'titles' to 'regulations' format."""
    if output_path is None:
        output_path = input_path
    
    logger.info(f"Reading CFR JSON from {input_path}...")
    
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Check current structure
        if "data" in data and "code_of_federal_regulations" in data["data"]:
            cfr_data = data["data"]["code_of_federal_regulations"]
            
            # Convert "titles" to "regulations" if it exists
            if "titles" in cfr_data:
                logger.info(f"Found {len(cfr_data['titles'])} regulations in 'titles' key")
                cfr_data["regulations"] = cfr_data.pop("titles")
                logger.info("Converted 'titles' key to 'regulations'")
            elif "regulations" in cfr_data:
                logger.info(f"JSON already uses 'regulations' key ({len(cfr_data['regulations'])} items)")
            else:
                logger.warning("Neither 'titles' nor 'regulations' key found in CFR data")
                return
        
        # Write converted data
        logger.info(f"Writing converted JSON to {output_path}...")
        temp_path = output_path.with_suffix('.json.tmp')
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        temp_path.replace(output_path)
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"Conversion complete! File size: {file_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"Error converting JSON: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent
    input_path = base_dir / "Data" / "Knowledge" / "cfr.json"
    
    if not input_path.exists():
        logger.error(f"CFR JSON file not found at {input_path}")
        exit(1)
    
    convert_cfr_json(input_path)

