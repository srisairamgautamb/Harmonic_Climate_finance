#!/bin/bash
set -e

echo "Creating conda environment: climate_finance_qml"
conda create -n climate_finance_qml python=3.10 -y
conda activate climate_finance_qml

echo "Installing requirements"
if [[ "$(uname -m)" == "arm64" ]]; then
    pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
    pip install jax==0.4.26 jaxlib==0.4.26 \
        -f https://storage.googleapis.com/jax-releases/jax_releases.html
    pip install -r requirements.txt --ignore-installed torch jax jaxlib
else
    pip install -r requirements.txt
fi

echo "Setting environment variables (add to your ~/.bashrc or .env file):"
echo "  export FRED_API_KEY=your_fred_api_key_here"
echo "  export NASDAQ_API_KEY=your_nasdaq_api_key_here"
echo "  Get FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html"

echo "Creating directory structure"
python -c "import main; main.create_directory_structure()"

echo "Environment ready."
echo "Run: python main.py --phases ALL"
echo "Or run single phase: python main.py --phases B1"
echo "Or skip download: python main.py --skip A2"
