#!/bin/bash

echo "🔧 Fixing yfinance issue..."
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Please run ./run.sh first"
    exit 1
fi

echo ""
echo "📥 Upgrading yfinance and dependencies..."
pip install --upgrade yfinance requests

echo ""
echo "✅ Fix applied!"
echo ""
echo "Now restart the Streamlit app:"
echo "  streamlit run app/streamlit_app.py"
