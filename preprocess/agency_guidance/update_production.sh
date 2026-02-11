#!/bin/bash
# Script to update production MongoDB with agency guidance documents and embeddings

cd "$(dirname "$0")"
cd ../../..

echo "========================================="
echo "Updating Production MongoDB"
echo "Agency Guidance with Embeddings"
echo "========================================="
echo ""

# Check if .env.production exists
if [ ! -f ".env.production" ] && [ ! -f "kyr-backend/.env.production" ]; then
    echo "ERROR: .env.production file not found!"
    echo "Please ensure .env.production exists in the project root or kyr-backend directory"
    exit 1
fi

echo "Running ingestion script with:"
echo "  - Production environment"
echo "  - Embeddings generation (batch mode)"
echo "  - Updating existing documents without embeddings"
echo ""

python kyr-backend/services/rag/preprocess/agency_guidance/ingest_agency_guidance.py \
    --production \
    --with-embeddings \
    --batch-embeddings

echo ""
echo "========================================="
echo "Update complete!"
echo "========================================="








































