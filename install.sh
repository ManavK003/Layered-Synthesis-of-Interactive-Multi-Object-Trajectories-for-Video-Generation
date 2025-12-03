#!/bin/bash

# Layered Multi-Object Video Generation - Installation Script

echo "=========================================="
echo "Installing Layered Video Generation"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (CUDA 11.8)
echo ""
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

# Install package in editable mode
echo ""
echo "Installing package..."
pip install -e .

# Create necessary directories
echo ""
echo "Creating directory structure..."
mkdir -p data/animal_kingdom
mkdir -p outputs
mkdir -p results
mkdir -p experiment_results
mkdir -p eval/metrics
mkdir -p src/models/layered_attention
mkdir -p src/models/trajectory
mkdir -p src/models/priors

# Download pretrained models (optional)
echo ""
echo "Downloading pretrained models..."
echo "This may take a while..."

python3 << END
from diffusers import DiffusionPipeline
import torch

print("Downloading Zeroscope model...")
pipeline = DiffusionPipeline.from_pretrained(
    "cerspense/zeroscope_v2_576w",
    torch_dtype=torch.float16
)
print("Model downloaded successfully!")
END

echo ""
echo "=========================================="
echo "✅ Installation complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the installation, run:"
echo "  python src/generate_multi_object.py --prompt 'A lion walking in the savanna' --output test.mp4"
echo ""