
#!/bin/bash
# setup.sh — environment setup for macOS/Linux

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt