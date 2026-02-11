# USCIS Policy Manual Update - Log Examples

## When Automated Update Runs

When the weekly update script runs and finds changes, you'll see logs like this:

### Weekly Update Script Logs (`weekly_update_uscis.py`)

```
[INFO] Starting weekly USCIS Policy Manual update check...
[INFO] Update will proceed: 7 days since last check (interval: 7 days)
[INFO] Starting USCIS Policy Manual update...
[INFO] Step 1: Downloading HTML...
[INFO] Download complete! File size: X.XX MB
[INFO] Step 2: Converting HTML to JSON...
[INFO] Converted HTML to JSON: ...
[INFO] Step 3: Loading JSON files...
[INFO] Current documents: 428, New documents: 450
[INFO] Step 4: Comparing documents...
[INFO] Changes detected:
[INFO]   - New documents: 5
[INFO]   - Updated documents: 12
[INFO]   - Deleted documents: 0
[INFO] Step 5: Updating MongoDB...
[INFO] Inserted 5 new documents
[INFO] Updated 12 existing documents
[INFO] Saved last update timestamp to ...
[INFO] Replaced current JSON with new JSON: ...
[INFO] Update complete!
[INFO] Update completed in 45.23 seconds
```

### When No Changes Are Detected

```
[INFO] Starting weekly USCIS Policy Manual update check...
[INFO] Update will proceed: 7 days since last check (interval: 7 days)
[INFO] Starting USCIS Policy Manual update...
[INFO] Step 1: Downloading HTML...
[INFO] Step 2: Converting HTML to JSON...
[INFO] Step 3: Loading JSON files...
[INFO] Current documents: 428, New documents: 428
[INFO] Step 4: Comparing documents...
[INFO] Changes detected:
[INFO]   - New documents: 0
[INFO]   - Updated documents: 0
[INFO]   - Deleted documents: 0
[INFO] No changes detected. MongoDB is up to date.
[INFO] Update complete!
[INFO] Update completed in 38.12 seconds
```

### When Update Is Skipped

```
[INFO] Starting weekly USCIS Policy Manual update check...
[INFO] Update skipped: Autoupdate is disabled
```

OR

```
[INFO] Starting weekly USCIS Policy Manual update check...
[INFO] Update skipped: Only 3 days since last check (interval: 7 days)
```

## Key Log Messages to Monitor

### ✅ Success Indicators

1. **"Update will proceed"** - Update is running
2. **"Inserted X new documents"** - New documents added
3. **"Updated X existing documents"** - Documents modified
4. **"Update complete!"** - Update finished successfully
5. **"Update completed in X.XX seconds"** - Total duration

### ⚠️ Warning Indicators

1. **"Found X deleted documents"** - Documents removed from source (not deleted from MongoDB)
2. **"BulkWriteError; inserted X docs"** - Partial insert success
3. **"Failed to update document"** - Individual document update failed

### ❌ Error Indicators

1. **"Failed to download HTML"** - Download failed
2. **"Error updating MongoDB"** - Database update failed
3. **"Error running update"** - Update script crashed

## MongoDB Update Details

When documents are updated, the logs show:

```
[INFO] Updated 12 existing documents
```

This means:
- 12 documents had their `updated_at` timestamp changed
- Content (text, references, clauses) was updated
- `created_at` was preserved (not changed)

## Search Result Validation Logs

When search results are validated against timestamps:

```
[INFO] [TIMESTAMP_VALIDATION] Removed 2 stale results from USCIS policy collection
[WARNING] [TIMESTAMP_VALIDATION] Removing stale result: document 507f1f77bcf86cd799439011 not found in MongoDB
```

## Monitoring Recommendations

1. **Check weekly update logs** for:
   - "Update will proceed" - Confirms update ran
   - "Updated X existing documents" - Shows how many were modified
   - "Update completed" - Confirms success

2. **Monitor for errors**:
   - Any ERROR level logs
   - WARNING about failed updates
   - Download failures

3. **Track update frequency**:
   - Should run weekly (every 7 days)
   - Check "days since last check" messages













