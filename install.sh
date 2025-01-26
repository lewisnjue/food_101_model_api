#!/bin/bash
set -e  # Exit immediately if any command fails
pip3 install --upgrade pip

pip3 install -r requirements.txt

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

