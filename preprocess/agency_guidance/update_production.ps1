# PowerShell script to update production MongoDB with agency guidance documents and embeddings

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
Set-Location ..\..\..

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Updating Production MongoDB" -ForegroundColor Cyan
Write-Host "Agency Guidance with Embeddings" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if .env.production exists
$envFile = ".env.production"
if (-not (Test-Path $envFile)) {
    $envFile = "kyr-backend\.env.production"
    if (-not (Test-Path $envFile)) {
        Write-Host "ERROR: .env.production file not found!" -ForegroundColor Red
        Write-Host "Please ensure .env.production exists in the project root or kyr-backend directory" -ForegroundColor Red
        exit 1
    }
}

Write-Host "Running ingestion script with:" -ForegroundColor Yellow
Write-Host "  - Production environment" -ForegroundColor Yellow
Write-Host "  - Embeddings generation (batch mode)" -ForegroundColor Yellow
Write-Host "  - Updating existing documents without embeddings" -ForegroundColor Yellow
Write-Host ""

python kyr-backend\services\rag\preprocess\agency_guidance\ingest_agency_guidance.py `
    --production `
    --with-embeddings `
    --batch-embeddings

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Ingestion failed!" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "Update complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green








































