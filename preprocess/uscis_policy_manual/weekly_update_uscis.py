# weekly_update_uscis.py
"""
Weekly update wrapper for USCIS Policy Manual.
Checks if autoupdate is enabled and if enough time has passed since last check.
"""
import argparse
import datetime
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

# Setup path for module execution
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root = backend_dir.parent

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import config
import importlib.util
import types

if 'backend' not in sys.modules:
    backend_mod = types.ModuleType('backend')
    sys.modules['backend'] = backend_mod

    services_init = backend_dir / 'services' / '__init__.py'
    if services_init.exists():
        spec = importlib.util.spec_from_file_location('backend.services', services_init)
        if spec and spec.loader:
            services_mod = importlib.util.module_from_spec(spec)
            sys.modules['backend.services'] = services_mod
            spec.loader.exec_module(services_mod)
            setattr(backend_mod, 'services', services_mod)

            rag_init = backend_dir / 'services' / 'rag' / '__init__.py'
            if rag_init.exists():
                spec = importlib.util.spec_from_file_location('backend.services.rag', rag_init)
                if spec and spec.loader:
                    rag_mod = importlib.util.module_from_spec(spec)
                    sys.modules['backend.services.rag'] = rag_mod
                    spec.loader.exec_module(rag_mod)
                    setattr(services_mod, 'rag', rag_mod)

                    config_file = backend_dir / 'services' / 'rag' / 'config.py'
                    if config_file.exists():
                        spec = importlib.util.spec_from_file_location('backend.services.rag.config', config_file)
                        if spec and spec.loader:
                            config_mod = importlib.util.module_from_spec(spec)
                            sys.modules['backend.services.rag.config'] = config_mod
                            spec.loader.exec_module(config_mod)
                            setattr(rag_mod, 'config', config_mod)

# Parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Weekly update for USCIS Policy Manual")
    parser.add_argument("--production", action="store_true", help="Use production environment")
    parser.add_argument("--dev", action="store_true", help="Use dev environment")
    parser.add_argument("--local", action="store_true", help="Use local environment")
    parser.add_argument("--force", action="store_true", help="Force update even if not enough time has passed")
    return parser.parse_args()

args = parse_args() if __name__ == "__main__" else None
env_override = None
if args:
    if args.production:
        env_override = "production"
    elif args.dev:
        env_override = "dev"
    elif args.local:
        env_override = "local"

# Import config
import backend.services.rag.config as config_module

if env_override:
    config_module.load_environment(env_override)

AUTOUPDATE_CONFIG = config_module.AUTOUPDATE_CONFIG
COLLECTION = config_module.COLLECTION
USCIS_POLICY_CONF = COLLECTION.get("USCIS_POLICY_SET")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("weekly_update_uscis")

BASE_DIR = Path(__file__).resolve().parents[2]

def check_autoupdate_enabled() -> bool:
    """Check if autoupdate is enabled."""
    # Check environment variable first
    env_enabled = os.getenv("USCIS_AUTOUPDATE_ENABLED", "").lower() == "true"
    if env_enabled:
        return True

    # Check AUTOUPDATE_CONFIG
    if AUTOUPDATE_CONFIG.get("enabled", False):
        return True

    # Check collection-specific config
    if USCIS_POLICY_CONF and USCIS_POLICY_CONF.get("autoupdate_enabled", False):
        return True

    return False

def should_run_update(force: bool = False) -> Tuple[bool, str]:
    """
    Check if update should run based on time since last check.

    Returns:
        Tuple of (should_run, reason)
    """
    if force:
        return True, "Force flag set"

    if not check_autoupdate_enabled():
        return False, "Autoupdate is disabled"

    # Check last check file
    last_check_file_path = AUTOUPDATE_CONFIG.get("last_check_file", "Data/Knowledge/.last_uscis_check")
    last_check_file = BASE_DIR / last_check_file_path
    check_interval_days = AUTOUPDATE_CONFIG.get("check_interval_days", 7)

    if not last_check_file.exists():
        return True, "No previous check found"

    try:
        with open(last_check_file, 'r') as f:
            last_check_str = f.read().strip()
        last_check = datetime.datetime.fromisoformat(last_check_str)
        days_since = (datetime.datetime.utcnow() - last_check).days

        if days_since >= check_interval_days:
            return True, f"{days_since} days since last check (interval: {check_interval_days} days)"
        else:
            return False, f"Only {days_since} days since last check (interval: {check_interval_days} days)"
    except Exception as e:
        logger.warning(f"Error reading last check file: {e}, proceeding with update")
        return True, "Error reading last check file"

def main():
    """Main function for weekly update."""
    logger.info("Starting weekly USCIS Policy Manual update check...")

    should_run, reason = should_run_update(force=args and args.force)

    if not should_run:
        logger.info(f"Update skipped: {reason}")
        return 0

    logger.info(f"Update will proceed: {reason}")

    # Import and run update script
    try:
        from update_uscis_policy_manual import main as update_main

        start_time = datetime.datetime.utcnow()
        result = update_main()
        end_time = datetime.datetime.utcnow()

        duration = (end_time - start_time).total_seconds()
        logger.info(f"Update completed in {duration:.2f} seconds")

        return result
    except Exception as e:
        logger.error(f"Error running update: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())

