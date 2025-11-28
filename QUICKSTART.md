# 🚀 Quick Start Guide - PriceVision

Get up and running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Internet connection (to download stock data)

## Installation

### Option 1: Automated Setup (Recommended)

#### On Linux/Mac:
```bash
./run.sh
```

#### On Windows:
```cmd
run.bat
```

The script will:
1. Create a virtual environment
2. Install all dependencies
3. Launch the Streamlit app

### Option 2: Manual Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd stock-trend-predictor
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment:**
   - Linux/Mac: `source venv/bin/activate`
   - Windows: `venv\Scripts\activate`

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the app:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

## First-Time Usage

1. **Access the app:** Open http://localhost:8501 in your browser

2. **Select a stock:**
   - Enter a ticker symbol (e.g., AAPL, MSFT, TSLA)
   - Or select from popular tickers dropdown

3. **Configure settings:**
   - Choose date range (recommend 5 years for better training)
   - Select model type (Random Forest is default)
   - Adjust model parameters if desired

4. **Train the model:**
   - Click "🚀 Train Model" button
   - Wait 10-60 seconds (depending on data size)
   - View model accuracy and metrics

5. **Make predictions:**
   - Click "🔮 Predict Next Day" button
   - See prediction with confidence scores
   - Explore visualizations in different tabs

6. **Download report:**
   - Go to "📄 Report" tab
   - Click "Generate PDF Report"
   - Download comprehensive PDF

## Common Issues

### Issue: ModuleNotFoundError
**Solution:** Make sure you activated the virtual environment and installed dependencies:
```bash
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Issue: yfinance download fails
**Solution:**
- Check your internet connection
- Try a different ticker symbol
- Ensure the ticker exists and is valid

### Issue: Streamlit won't start
**Solution:**
```bash
pip install --upgrade streamlit
streamlit run app/streamlit_app.py
```

### Issue: Model training is slow
**Solution:**
- Reduce the date range (use fewer years)
- Reduce number of estimators in settings
- This is normal for large datasets

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_features.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
stock-trend-predictor/
├── app/                    # Streamlit web app
├── src/                    # Core Python modules
├── models/                 # Saved model files
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks
├── requirements.txt        # Python dependencies
└── README.md              # Full documentation
```

## Next Steps

- Experiment with different stocks
- Try different model parameters
- Explore the Jupyter notebook in `notebooks/`
- Read the full README.md for detailed documentation

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/stock-trend-predictor/issues)
- Documentation: [README.md](README.md)

## Remember

⚠️ **This is for educational purposes only. Not financial advice!**

Happy learning! 🎓
