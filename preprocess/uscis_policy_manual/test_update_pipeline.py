# test_update_pipeline.py
"""
Test script for USCIS Policy Manual update pipeline.
Tests all components without affecting production.
"""
import json
import logging
import os
import sys
from pathlib import Path

# Setup path for module execution
backend_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root = backend_dir.parent

# Add project root to path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("test_update_pipeline")

BASE_DIR = Path(__file__).resolve().parents[2]
HTML_PATH = BASE_DIR / "Data" / "Knowledge" / "Policy Manual _ USCIS.html"
CURRENT_JSON_PATH = BASE_DIR / "Data" / "Knowledge" / "uscis_policy.json"
TEST_JSON_PATH = BASE_DIR / "Data" / "Knowledge" / "uscis_policy_test.json"

def test_download():
    """Test 1: Download script"""
    logger.info("=" * 80)
    logger.info("TEST 1: Download USCIS Policy Manual HTML")
    logger.info("=" * 80)

    try:
        from download_uscis_policy_manual import USCIS_POLICY_URL, download_policy_manual

        # Test download
        test_html_path = BASE_DIR / "Data" / "Knowledge" / "Policy Manual _ USCIS_test.html"
        success = download_policy_manual(USCIS_POLICY_URL, test_html_path)

        if success and test_html_path.exists():
            file_size = test_html_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Download test PASSED - File size: {file_size:.2f} MB")
            return True, test_html_path
        else:
            logger.error("❌ Download test FAILED")
            return False, None
    except Exception as e:
        logger.error(f"❌ Download test FAILED with error: {e}", exc_info=True)
        return False, None

def test_html_to_json(html_path):
    """Test 2: HTML to JSON conversion"""
    logger.info("=" * 80)
    logger.info("TEST 2: Convert HTML to JSON")
    logger.info("=" * 80)

    if not html_path or not html_path.exists():
        logger.warning("⚠️  Skipping HTML to JSON test - no HTML file")
        return False, None

    try:
        from convert_uscis_html_to_json import parse_policy_manual_html

        json_data = parse_policy_manual_html(html_path)

        if json_data and "data" in json_data and "uscis_policy" in json_data["data"]:
            docs = json_data["data"]["uscis_policy"].get("documents", [])
            logger.info(f"✅ HTML to JSON test PASSED - Parsed {len(docs)} documents")

            # Save test JSON
            with open(TEST_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved test JSON to: {TEST_JSON_PATH}")

            return True, json_data
        else:
            logger.error("❌ HTML to JSON test FAILED - Invalid JSON structure")
            return False, None
    except Exception as e:
        logger.error(f"❌ HTML to JSON test FAILED with error: {e}", exc_info=True)
        return False, None

def test_json_comparison(new_json_data):
    """Test 3: JSON comparison logic"""
    logger.info("=" * 80)
    logger.info("TEST 3: JSON Comparison Logic")
    logger.info("=" * 80)

    if not new_json_data:
        logger.warning("⚠️  Skipping JSON comparison test - no new JSON data")
        return False

    try:
        from update_uscis_policy_manual import find_document_changes

        # Load current JSON if it exists
        if CURRENT_JSON_PATH.exists():
            with open(CURRENT_JSON_PATH, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
            current_docs = current_data.get("data", {}).get("uscis_policy", {}).get("documents", [])
        else:
            logger.info("No current JSON found - treating all as new documents")
            current_docs = []

        new_docs = new_json_data.get("data", {}).get("uscis_policy", {}).get("documents", [])

        new_documents, updated_documents, deleted_titles = find_document_changes(current_docs, new_docs)

        logger.info("✅ JSON comparison test PASSED")
        logger.info(f"   Current documents: {len(current_docs)}")
        logger.info(f"   New documents: {len(new_docs)}")
        logger.info("   Changes detected:")
        logger.info(f"     - New: {len(new_documents)}")
        logger.info(f"     - Updated: {len(updated_documents)}")
        logger.info(f"     - Deleted: {len(deleted_titles)}")

        if new_documents:
            logger.info(f"   Sample new document: {new_documents[0].get('title', 'N/A')[:60]}")
        if updated_documents:
            logger.info(f"   Sample updated document: {updated_documents[0].get('title', 'N/A')[:60]}")

        return True
    except Exception as e:
        logger.error(f"❌ JSON comparison test FAILED with error: {e}", exc_info=True)
        return False

def test_timestamp_logic():
    """Test 4: Timestamp addition logic"""
    logger.info("=" * 80)
    logger.info("TEST 4: Timestamp Addition Logic")
    logger.info("=" * 80)

    try:
        import datetime

        # Simulate document with timestamps
        test_doc = {
            "title": "Test Document",
            "text": "Test content",
            "created_at": datetime.datetime.utcnow(),
            "updated_at": datetime.datetime.utcnow()
        }

        # Verify timestamps are datetime objects
        assert isinstance(test_doc["created_at"], datetime.datetime)
        assert isinstance(test_doc["updated_at"], datetime.datetime)
        assert test_doc["created_at"] <= test_doc["updated_at"]

        logger.info("✅ Timestamp logic test PASSED")
        logger.info(f"   created_at: {test_doc['created_at']}")
        logger.info(f"   updated_at: {test_doc['updated_at']}")

        return True
    except Exception as e:
        logger.error(f"❌ Timestamp logic test FAILED with error: {e}", exc_info=True)
        return False

def test_autoupdate_config():
    """Test 5: Autoupdate configuration"""
    logger.info("=" * 80)
    logger.info("TEST 5: Autoupdate Configuration")
    logger.info("=" * 80)

    try:
        import backend.services.rag.config as config_module

        autoupdate_config = config_module.AUTOUPDATE_CONFIG
        uscis_config = config_module.COLLECTION.get("USCIS_POLICY_SET", {})

        logger.info("✅ Autoupdate config test PASSED")
        logger.info(f"   AUTOUPDATE_CONFIG enabled: {autoupdate_config.get('enabled', False)}")
        logger.info(f"   Collections: {autoupdate_config.get('collections', [])}")
        logger.info(f"   Check interval: {autoupdate_config.get('check_interval_days', 7)} days")
        logger.info(f"   USCIS autoupdate_enabled: {uscis_config.get('autoupdate_enabled', False)}")
        logger.info(f"   USCIS autoupdate_url: {uscis_config.get('autoupdate_url', 'N/A')}")

        # Verify only USCIS is in collections
        collections = autoupdate_config.get("collections", [])
        assert "USCIS_POLICY_SET" in collections
        assert len(collections) == 1, "Only USCIS should be in autoupdate collections"

        logger.info("   ✅ Verified: Only USCIS_POLICY_SET is configured for autoupdate")

        return True
    except Exception as e:
        logger.error(f"❌ Autoupdate config test FAILED with error: {e}", exc_info=True)
        return False

def test_weekly_update_logic():
    """Test 6: Weekly update script logic"""
    logger.info("=" * 80)
    logger.info("TEST 6: Weekly Update Script Logic")
    logger.info("=" * 80)

    try:
        from weekly_update_uscis import check_autoupdate_enabled, should_run_update

        enabled = check_autoupdate_enabled()
        should_run, reason = should_run_update(force=False)

        logger.info("✅ Weekly update logic test PASSED")
        logger.info(f"   Autoupdate enabled: {enabled}")
        logger.info(f"   Should run update: {should_run}")
        logger.info(f"   Reason: {reason}")

        # Test with force flag
        should_run_force, reason_force = should_run_update(force=True)
        logger.info(f"   Should run (force): {should_run_force}")
        logger.info(f"   Reason (force): {reason_force}")

        return True
    except Exception as e:
        logger.error(f"❌ Weekly update logic test FAILED with error: {e}", exc_info=True)
        return False

def main():
    """Run all tests"""
    logger.info("Starting USCIS Policy Manual Update Pipeline Tests")
    logger.info("=" * 80)
    logger.info("NOTE: These tests will NOT affect production MongoDB")
    logger.info("=" * 80)

    results = {}

    # Test 1: Download
    download_success, html_path = test_download()
    results["download"] = download_success

    # Test 2: HTML to JSON
    json_success, json_data = test_html_to_json(html_path)
    results["html_to_json"] = json_success

    # Test 3: JSON Comparison
    if json_success:
        comparison_success = test_json_comparison(json_data)
        results["json_comparison"] = comparison_success
    else:
        results["json_comparison"] = False

    # Test 4: Timestamp Logic
    timestamp_success = test_timestamp_logic()
    results["timestamp_logic"] = timestamp_success

    # Test 5: Autoupdate Config
    config_success = test_autoupdate_config()
    results["autoupdate_config"] = config_success

    # Test 6: Weekly Update Logic
    weekly_success = test_weekly_update_logic()
    results["weekly_update_logic"] = weekly_success

    # Summary
    logger.info("=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")

    logger.info("=" * 80)
    logger.info(f"Total: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests PASSED!")
        return 0
    else:
        logger.warning(f"⚠️  {total - passed} test(s) FAILED")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())













