# 📈 PriceVision - AI Stock Trend Predictor

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An intelligent stock price trend prediction application powered by machine learning. PriceVision analyzes historical stock data and predicts next-day price trends (UP, DOWN, or FLAT) with confidence scores.

> ⚠️ **EDUCATIONAL PURPOSE ONLY** - This application is designed for educational and portfolio demonstration purposes. It does NOT constitute financial advice. Do not use these predictions for actual trading or investment decisions.

---

## 🎯 Features

- **📊 Trend Prediction**: Predict next-day stock price movements (UP/DOWN/FLAT)
- **🤖 Machine Learning**: Uses Random Forest or Gradient Boosting classifiers
- **📈 Technical Indicators**: 15+ engineered features including moving averages, RSI, volatility
- **🎨 Interactive Visualizations**: Beautiful charts for price history, confusion matrices, feature importance
- **📊 Model Explainability**: Feature importance analysis and performance metrics
- **📄 PDF Reports**: Downloadable comprehensive prediction reports
- **🌐 Web Interface**: Easy-to-use Streamlit dashboard
- **✅ Tested**: Comprehensive unit test coverage

---

## 🛠️ Tech Stack

- **Python 3.8+**
- **Machine Learning**: scikit-learn (Random Forest, Gradient Boosting)
- **Data Processing**: pandas, numpy
- **Data Source**: yfinance (Yahoo Finance API)
- **Visualization**: matplotlib, plotly, seaborn
- **Web Framework**: Streamlit
- **PDF Generation**: ReportLab
- **Testing**: pytest

---

## 📁 Project Structure

```
stock-trend-predictor/
│
├── app/
│   └── streamlit_app.py          # Main Streamlit application
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Stock data fetching (yfinance)
│   ├── features.py                # Feature engineering & labels
│   ├── model.py                   # Model training & evaluation
│   ├── predict.py                 # Prediction engine
│   ├── visualize.py               # Charting & visualization
│   ├── report.py                  # PDF report generation
│   └── utils.py                   # Utility functions
│
├── models/                        # Saved model files (.pkl)
├── notebooks/                     # Jupyter notebooks for experiments
├── tests/                         # Unit tests
│   ├── test_features.py
│   ├── test_model.py
│   └── test_predict.py
│
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stock-trend-predictor.git
cd stock-trend-predictor
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`

### Using the Application

1. **Configure Settings** (Left Sidebar):
   - Enter a stock ticker (e.g., AAPL, MSFT, TSLA)
   - Select date range (1-10 years of historical data)
   - Choose model type and parameters

2. **Train Model**:
   - Click "🚀 Train Model" button
   - Wait for data download and model training to complete
   - View model accuracy and performance metrics

3. **Make Prediction**:
   - Click "🔮 Predict Next Day" button
   - See prediction with confidence scores
   - Explore detailed analysis in different tabs

4. **Download Report**:
   - Navigate to "📄 Report" tab
   - Generate and download comprehensive PDF report

---

## 📊 How It Works

### 1. Data Collection
- Downloads historical OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance
- Supports customizable date ranges (1-10 years)

### 2. Feature Engineering
Creates 15+ technical indicators:
- **Price Features**: Daily returns, moving averages (5, 10, 20-day)
- **Volatility**: Rolling standard deviation of returns
- **Volume**: Volume changes and ratios
- **Position**: Price relative to moving averages
- **Momentum**: RSI (Relative Strength Index)
- **Lag Features**: Previous day values

### 3. Label Creation
Defines trend based on next-day price movement:
- **UP** ⬆️: Next day close ≥ today's close × 1.005 (+0.5%)
- **DOWN** ⬇️: Next day close ≤ today's close × 0.995 (-0.5%)
- **FLAT** ➡️: Between -0.5% and +0.5%

### 4. Model Training
- **Train/Test Split**: Chronological (80/20) - NO random shuffling
- **Algorithms**: Random Forest (default) or Gradient Boosting
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### 5. Prediction
- Processes latest data through feature pipeline
- Outputs trend prediction with probability distribution
- Categorizes confidence level (HIGH/MEDIUM/LOW)

---

## 📈 Model Performance

Typical performance metrics on 5 years of data:

- **Accuracy**: 50-65% (significantly better than random 33%)
- **Precision**: Varies by class (UP/DOWN/FLAT)
- **Feature Importance**: Moving averages and momentum indicators typically most important

> **Note**: Stock market prediction is inherently uncertain. No model can consistently predict market movements with high accuracy.

---

## 🧪 Testing

Run unit tests:

```bash
pytest tests/ -v
```

Run specific test file:

```bash
pytest tests/test_features.py -v
```

Test coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

---

## 🎓 Educational Value

This project demonstrates:

✅ **Full ML Pipeline**: Data acquisition → Feature engineering → Training → Evaluation → Deployment
✅ **Time Series Handling**: Proper chronological train/test splits
✅ **Feature Engineering**: Creating meaningful technical indicators
✅ **Model Evaluation**: Comprehensive metrics and visualization
✅ **Production Deployment**: Web app with user-friendly interface
✅ **Code Quality**: Modular design, type hints, documentation, tests
✅ **Best Practices**: Clean code, separation of concerns, error handling

---

## 📸 Screenshots

### Main Dashboard
![Dashboard](screenshots/dashboard.png)
*Interactive dashboard with prediction and visualization*

### Prediction Results
![Prediction](screenshots/prediction.png)
*Next-day trend prediction with confidence scores*

### Model Performance
![Performance](screenshots/performance.png)
*Confusion matrix and feature importance*

> **Note**: Screenshots are placeholders - add actual screenshots after running the app

---

## 🔮 Future Enhancements

Potential improvements (not in current version):

- [ ] LSTM/Transformer models for sequence learning
- [ ] Sentiment analysis from news headlines
- [ ] Portfolio simulation and backtesting
- [ ] Multi-stock correlation analysis
- [ ] Real-time data streaming
- [ ] API endpoint for programmatic access
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure/GCP)

---

## ⚠️ Disclaimer

**IMPORTANT**: This application is for **educational purposes only**.

- ❌ **NOT financial advice**
- ❌ **NOT a guarantee of profitability**
- ❌ **NOT suitable for real trading**
- ✅ **FOR learning and demonstration only**

Stock market predictions are inherently uncertain. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Hirko Joba**

- Portfolio: [Your Portfolio URL]
- LinkedIn: [Your LinkedIn]
- GitHub: [@yourusername](https://github.com/yourusername)

---

## 🙏 Acknowledgments

- **yfinance**: For providing free access to Yahoo Finance data
- **scikit-learn**: For excellent ML algorithms
- **Streamlit**: For making web app development incredibly easy
- **The ML Community**: For countless tutorials and resources

---

## 📚 Resources

- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Technical Analysis Indicators](https://www.investopedia.com/technical-analysis-4689657)

---

## 🤝 Contributing

This is a portfolio project, but suggestions and feedback are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 💬 Questions?

If you have questions or feedback:

- Open an [Issue](https://github.com/yourusername/stock-trend-predictor/issues)
- Contact me via [LinkedIn](your-linkedin-url)

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

Made with ❤️ for learning and education

</div>
