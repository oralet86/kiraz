#!/bin/bash

# Exit on any error
set -e

echo "--- Starting System Update & Upgrade ---"
sudo apt update && sudo apt upgrade -y

echo "--- Verifying NVIDIA Driver Status ---"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "WARNING: nvidia-smi not found. Ensure your GPU drivers are installed."
fi

echo "--- Installing Pyenv and Python Build Dependencies ---"
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
libffi-dev liblzma-dev

echo "--- Installing Pyenv ---"
if [ -d "$HOME/.pyenv" ]; then
    echo "Pyenv already installed. Skipping installation."
else
    curl https://pyenv.run | bash
fi

echo "--- Setting up Pyenv Environment ---"
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"

echo "--- Installing Python 3.14.3 ---"
if pyenv versions | grep -q "3.14.3"; then
    echo "Python 3.14.3 already installed. Skipping installation."
else
    pyenv install 3.14.3
fi

echo "--- Setting Local Python Version ---"
pyenv local 3.14.3

echo "--- Installing OpenCV System Dependencies ---"
sudo apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev

echo "--- Creating Virtual Environment in './.venv' ---"
# Check if venv already exists to avoid overwriting
if [ -d ".venv" ]; then
    echo "Directory '.venv' already exists. Skipping creation."
else
    python3 -m venv .venv
    echo "Virtual environment '.venv' created successfully."
fi

echo "--- Installing Python Dependencies ---"
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "--- Installing gdown for Google Drive Downloads ---"
pip install gdown

echo "--- Downloading Datasets ---"
if [ -f ".env" ]; then
    # Extract GOOGLE_DRIVE_DATASET_URL from .env (skip comments and empty lines)
    GOOGLE_DRIVE_URL=$(grep "^GOOGLE_DRIVE_DATASET_URL=" .env | cut -d'=' -f2)
    
    if [ ! -z "$GOOGLE_DRIVE_URL" ]; then
        echo "Downloading datasets from Google Drive..."
        gdown --fuzzy "$GOOGLE_DRIVE_URL" --output datasets.zip
        
        if [ -f "datasets.zip" ]; then
            echo "Extracting datasets..."
            unzip -q datasets.zip
            rm datasets.zip
            echo "Datasets setup complete!"
        else
            echo "Failed to download datasets.zip"
        fi
    else
        echo "GOOGLE_DRIVE_DATASET_URL not found in .env file"
    fi
else
    echo "No .env file found. Skipping dataset download."
    echo "Create .env file with: GOOGLE_DRIVE_DATASET_URL=https://drive.google.com/uc?id=YOUR_FILE_ID"
fi

echo "--- Setup Complete ---"
echo "To activate your environment, run: source .venv/bin/activate"
echo "Python version: $(python --version)"