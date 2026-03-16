# download_uscis_policy_manual.py
"""
Download USCIS Policy Manual HTML from https://www.uscis.gov/policy-manual
"""
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import requests

# Setup path for module execution
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root = backend_dir.parent

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("download_uscis_policy")

USCIS_POLICY_URL = "https://www.uscis.gov/policy-manual"
BASE_DIR = Path(__file__).resolve().parents[2]
HTML_OUTPUT_PATH = BASE_DIR / "Data" / "Knowledge" / "Policy Manual _ USCIS.html"

def download_policy_manual(url: str, output_path: Path, max_retries: int = 3, retry_delay: int = 5) -> bool:
    """
    Download USCIS Policy Manual HTML from the given URL.

    Args:
        url: URL to download from
        output_path: Path to save the HTML file
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries

    Returns:
        True if download successful, False otherwise
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading USCIS Policy Manual from {url} (attempt {attempt}/{max_retries})...")

            response = requests.get(url, headers=headers, timeout=60, stream=True)
            response.raise_for_status()

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first, then rename (atomic operation)
            temp_path = output_path.with_suffix('.html.tmp')
            total_size = 0

            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)

            # Rename temp file to final file (atomic)
            temp_path.replace(output_path)

            file_size_mb = total_size / (1024 * 1024)
            logger.info(f"Download complete! File size: {file_size_mb:.2f} MB")
            logger.info(f"Saved to: {output_path}")

            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"Download attempt {attempt} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to download after {max_retries} attempts")
                return False
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}", exc_info=True)
            return False

    return False

def main():
    """Main function to download USCIS Policy Manual."""
    logger.info("Starting USCIS Policy Manual download...")

    # Get URL from config if available, otherwise use default
    try:
        # Try to import config to get URL
        import importlib.util
        config_path = backend_dir / 'services' / 'rag' / 'config.py'
        if config_path.exists():
            spec = importlib.util.spec_from_file_location("config", config_path)
            if spec and spec.loader:
                config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config)
                uscis_config = config.COLLECTION.get("USCIS_POLICY_SET", {})
                url = uscis_config.get("autoupdate_url", USCIS_POLICY_URL)
                logger.info(f"Using URL from config: {url}")
            else:
                url = USCIS_POLICY_URL
        else:
            url = USCIS_POLICY_URL
    except Exception as e:
        logger.warning(f"Could not load config, using default URL: {e}")
        url = USCIS_POLICY_URL

    success = download_policy_manual(url, HTML_OUTPUT_PATH)

    if success:
        logger.info("USCIS Policy Manual download completed successfully")
        return 0
    else:
        logger.error("USCIS Policy Manual download failed")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())













