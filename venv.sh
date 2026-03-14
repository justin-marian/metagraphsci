#!bin/bash

echo -en "Setting up the environment...\n\n"
python3 -m venv .venv
source .venv/bin/activate

pip install -U -r requirements.txt
pip install -r requirements.txt
echo -en "Setup complete.\n\n"

echo -en "Installing PyTorch with CUDA support...\n\n"
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
