# DeepLabCut Environment Setup Script
# Uses UV package manager for fast, isolated installation

Write-Host "🐄 Setting up DeepLabCut environment..." -ForegroundColor Cyan

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "❌ UV not found. Installing UV..." -ForegroundColor Yellow
    irm https://astral.sh/uv/install.ps1 | iex
    
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

Write-Host "✅ UV found" -ForegroundColor Green

# Create virtual environment
Write-Host "`n📦 Creating virtual environment..." -ForegroundColor Green
uv venv .venv

if (-not $?) {
    Write-Host "❌ Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Activate environment
Write-Host "🔌 Activating environment..." -ForegroundColor Green
& .\.venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "`n⬇️  Installing dependencies (this may take 5-10 minutes)..." -ForegroundColor Green
Write-Host "   This includes DeepLabCut 2.3.9 and NumPy 1.23.5..." -ForegroundColor Yellow

uv pip install -r requirements.txt

if (-not $?) {
    Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Verify installation
Write-Host "`n✅ Verifying installation..." -ForegroundColor Green

python -c "import deeplabcut; print(f'✅ DeepLabCut {deeplabcut.__version__} installed successfully')"
python -c "import numpy; print(f'✅ NumPy {numpy.__version__}')"
python -c "import tensorflow as tf; print(f'✅ TensorFlow {tf.__version__}')"

Write-Host "`n🎉 Setup complete!" -ForegroundColor Cyan
Write-Host "   To activate environment: .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "   To run test mode: python process_videos.py --test" -ForegroundColor White
Write-Host "   To run batch mode: python process_videos.py --batch" -ForegroundColor White
