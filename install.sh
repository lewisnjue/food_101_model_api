#/bin/bash
python3 -m venv env
./env/bin/pip install -r requirements.txt

./env/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

