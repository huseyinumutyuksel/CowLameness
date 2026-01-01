# MMPose Environment Setup Script
# Uses UV package manager for fast installation

Write-Host "Setting up MMPose environment..." -ForegroundColor Cyan

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "UV not found. Installing UV..." -ForegroundColor Yellow
    irm https://astral.sh/uv/install.ps1 | iex
    
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' + [System.Environment]::GetEnvironmentVariable('Path','User')
}

Write-Host "UV found" -ForegroundColor Green

# Create virtual environment
Write-Host "`nCreating virtual environment..." -ForegroundColor Green
uv venv .venv

if (-not $?) {
    Write-Host "Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Activate environment
Write-Host "Activating environment..." -ForegroundColor Green
& .\.venv\Scripts\Activate.ps1

# Install PyTorch first (for CUDA support if available)
Write-Host "`nInstalling PyTorch..." -ForegroundColor Green
uv pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
Write-Host "`nInstalling MMPose and dependencies (this may take 5-10 minutes)..." -ForegroundColor Green
uv pip install -r requirements.txt

if (-not $?) {
    Write-Host "Failed to install dependencies" -ForegroundColor Red
    exit 1
}

# Verify installation
Write-Host "`nVerifying installation..." -ForegroundColor Green

python -c "import mmpose; print(f'MMPose {mmpose.__version__} installed successfully')"
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import mmcv; print(f'MMCV {mmcv.__version__}')"

Write-Host "`nSetup complete!" -ForegroundColor Cyan
Write-Host "To activate environment: .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "To run test mode: python process_videos.py --test" -ForegroundColor White
Write-Host "To run batch mode: python process_videos.py --batch" -ForegroundColor White
