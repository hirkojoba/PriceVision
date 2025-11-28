"""
PriceVision - AI Stock Trend Predictor
Streamlit Web Application
"""

import streamlit as st
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import tempfile

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_stock_data, validate_date_range
from src.features import prepare_dataset, add_technical_indicators
from src.model import (
    train_and_evaluate_pipeline,
    load_model,
    get_feature_importance,
    save_model
)
from src.predict import predict_with_context, get_prediction_confidence_level
from src.visualize import (
    plot_price_history,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_confidence_bars,
    plot_metrics_summary
)
from src.report import generate_pdf_report
from src.utils import (
    get_default_date_range,
    validate_ticker,
    get_model_path,
    format_currency,
    format_percentage,
    get_trend_emoji,
    get_trend_color
)

# Page configuration
st.set_page_config(
    page_title="PriceVision - AI Stock Trend Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        color: #856404;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None
if 'df' not in st.session_state:
    st.session_state.df = None


def main():
    """Main application function."""

    # Header
    st.markdown('<div class="main-header">📈 PriceVision</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Stock Trend Predictor</div>', unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>EDUCATIONAL PURPOSE ONLY</strong><br>
        This application is designed for educational and demonstration purposes.
        It does NOT constitute financial advice. Do not use these predictions for actual trading decisions.
        Always consult with a qualified financial advisor before making investment decisions.
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Ticker input
        ticker_input = st.text_input(
            "Stock Ticker Symbol",
            value="AAPL",
            help="Enter a valid stock ticker (e.g., AAPL, MSFT, TSLA)"
        ).strip().upper()

        # Quick select popular tickers
        popular_tickers = st.selectbox(
            "Or select a popular ticker:",
            ["", "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"]
        )

        if popular_tickers:
            ticker_input = popular_tickers

        # Date range
        st.subheader("📅 Data Range")
        default_start, default_end = get_default_date_range(5)

        years_back = st.slider(
            "Years of historical data",
            min_value=1,
            max_value=10,
            value=5,
            help="More data = better training, but slower"
        )

        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)

        col1, col2 = st.columns(2)
        with col1:
            start_input = st.date_input(
                "Start Date",
                value=start_date,
                max_value=datetime.now()
            )
        with col2:
            end_input = st.date_input(
                "End Date",
                value=end_date,
                max_value=datetime.now()
            )

        # Model parameters
        st.subheader("Model Settings")
        model_type = st.selectbox(
            "Model Type",
            ["Random Forest", "Gradient Boosting"],
            help="Random Forest is generally more stable"
        )

        n_estimators = st.slider(
            "Number of Estimators",
            min_value=50,
            max_value=500,
            value=200,
            step=50,
            help="More estimators = better performance but slower training"
        )

        test_size = st.slider(
            "Test Set Size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            help="Proportion of data used for testing"
        )

        st.markdown("---")

        # Action buttons
        train_button = st.button("Train Model", use_container_width=True, type="primary")
        predict_button = st.button("Predict Next Day", use_container_width=True)

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🎯 Prediction",
        "📈 Model Performance",
        "📄 Report"
    ])

    # Training logic
    if train_button:
        try:
            # Validate ticker
            ticker = validate_ticker(ticker_input)
            st.session_state.current_ticker = ticker

            with st.spinner(f"📥 Downloading data for {ticker}..."):
                # Load data
                df = load_stock_data(
                    ticker,
                    start_input.strftime('%Y-%m-%d'),
                    end_input.strftime('%Y-%m-%d')
                )
                st.session_state.df = df

            st.success(f"✅ Downloaded {len(df)} days of data")

            with st.spinner("🔧 Engineering features..."):
                # Prepare dataset
                X, y = prepare_dataset(df)
                st.session_state.feature_names = X.columns.tolist()

            st.success(f"✅ Created {len(X.columns)} features")

            with st.spinner(f"🤖 Training {model_type} model..."):
                # Train model
                model_type_key = "random_forest" if model_type == "Random Forest" else "gradient_boosting"

                model, metrics = train_and_evaluate_pipeline(
                    X, y,
                    test_size=test_size,
                    model_type=model_type_key,
                    n_estimators=n_estimators
                )

                # Save model
                model_path = get_model_path(ticker)
                save_model(model, model_path, st.session_state.feature_names)

                st.session_state.model = model
                st.session_state.metrics = metrics
                st.session_state.model_type = model_type
                st.session_state.model_trained = True

            st.success(f"✅ {model_type} trained! Accuracy: {metrics['accuracy']:.2%}")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.model_trained = False

    # Prediction logic
    if predict_button:
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train a model first!")
        else:
            try:
                ticker = st.session_state.current_ticker

                with st.spinner("🔮 Making prediction..."):
                    # Reload recent data
                    df_recent = load_stock_data(
                        ticker,
                        (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                        datetime.now().strftime('%Y-%m-%d')
                    )

                    # Add features
                    df_with_features = add_technical_indicators(df_recent)

                    # Make prediction
                    prediction = predict_with_context(
                        st.session_state.model,
                        df_with_features,
                        st.session_state.feature_names
                    )

                    st.session_state.prediction = prediction
                    st.session_state.prediction_made = True

                st.success("✅ Prediction complete!")

            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")

    # TAB 1: Overview
    with tab1:
        st.header("📊 Data Overview")

        if st.session_state.df is not None:
            df = st.session_state.df

            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Days", len(df))
            with col2:
                st.metric("Latest Price", format_currency(df['Close'].iloc[-1]))
            with col3:
                price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0])
                st.metric("Total Return", format_percentage(price_change))
            with col4:
                st.metric("Avg Volume", f"{df['Volume'].mean():,.0f}")

            # Price chart
            st.subheader("Price History")
            if 'ma_5' in df.columns:
                fig = plot_price_history(df, st.session_state.current_ticker, show_ma=True)
            else:
                df_with_ma = add_technical_indicators(df)
                fig = plot_price_history(df_with_ma, st.session_state.current_ticker, show_ma=True)

            st.pyplot(fig)

            # Data preview
            with st.expander("📋 View Raw Data"):
                st.dataframe(df.tail(20), use_container_width=True)

        else:
            st.info("👆 Configure settings in the sidebar and click 'Train Model' to get started")

    # TAB 2: Prediction
    with tab2:
        st.header("🔮 Next-Day Trend Prediction")

        if st.session_state.prediction_made and st.session_state.prediction:
            pred = st.session_state.prediction
            trend = pred['trend']
            confidence = pred['confidence']

            # Prediction display
            trend_color = get_trend_color(trend)
            trend_emoji = get_trend_emoji(trend)

            st.markdown(f"""
            <div class="prediction-box" style="background-color: {trend_color}20; border: 2px solid {trend_color};">
                <h1 style="color: {trend_color}; margin: 0;">{trend_emoji} {trend}</h1>
                <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                    Confidence Level: <strong>{get_prediction_confidence_level(confidence)}</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Confidence breakdown
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Confidence Distribution")
                fig = plot_confidence_bars(confidence)
                st.pyplot(fig)

            with col2:
                st.subheader("Prediction Details")
                st.metric("Latest Close Price", format_currency(pred['latest_close']))
                st.metric("Latest Date", pred['latest_date'])
                st.metric("Prediction For", pred['next_day_prediction_date'])

                st.markdown("#### Probabilities")
                for trend_name, prob in sorted(confidence.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"**{trend_name}:** {format_percentage(prob)}")

        else:
            st.info("👆 Train a model and click 'Predict Next Day' to see predictions")

    # TAB 3: Model Performance
    with tab3:
        st.header("📈 Model Performance Analysis")

        if st.session_state.metrics:
            metrics = st.session_state.metrics

            # Show model type
            if 'model_type' in st.session_state and st.session_state.model_type:
                st.info(f"🤖 Model Type: **{st.session_state.model_type}**")

            # Overall metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Overall Accuracy", format_percentage(metrics['accuracy']))
            with col2:
                avg_precision = sum(metrics['precision'].values()) / 3
                st.metric("Avg Precision", format_percentage(avg_precision))
            with col3:
                avg_recall = sum(metrics['recall'].values()) / 3
                st.metric("Avg Recall", format_percentage(avg_recall))

            # Detailed metrics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Confusion Matrix")
                fig_cm = plot_confusion_matrix(metrics['confusion_matrix'])
                st.pyplot(fig_cm)

            with col2:
                st.subheader("Per-Class Metrics")
                fig_metrics = plot_metrics_summary(metrics)
                st.pyplot(fig_metrics)

            # Feature importance
            if st.session_state.model:
                st.subheader("Feature Importance")
                fig_importance = plot_feature_importance(
                    st.session_state.model,
                    st.session_state.feature_names,
                    top_n=15
                )
                st.pyplot(fig_importance)

            # Classification report
            with st.expander("📊 Detailed Classification Report"):
                st.text(metrics['classification_report'])

        else:
            st.info("👆 Train a model to see performance metrics")

    # TAB 4: Report
    with tab4:
        st.header("📄 Download Report")

        if st.session_state.model_trained and st.session_state.prediction_made:
            st.write("Generate a comprehensive PDF report with all predictions and metrics.")

            if st.button("📥 Generate PDF Report", type="primary"):
                try:
                    with st.spinner("Generating PDF report..."):
                        # Create temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            pdf_path = tmp_file.name

                        # Generate report
                        generate_pdf_report(
                            ticker=st.session_state.current_ticker,
                            start_date=start_input.strftime('%Y-%m-%d'),
                            end_date=end_input.strftime('%Y-%m-%d'),
                            metrics=st.session_state.metrics,
                            prediction=st.session_state.prediction,
                            output_path=pdf_path
                        )

                        # Read file
                        with open(pdf_path, 'rb') as f:
                            pdf_data = f.read()

                        # Download button
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_data,
                            file_name=f"{st.session_state.current_ticker}_prediction_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )

                        st.success("✅ Report generated successfully!")

                        # Cleanup
                        os.unlink(pdf_path)

                except Exception as e:
                    st.error(f"❌ Error generating report: {str(e)}")

        else:
            st.info("👆 Train a model and make a prediction first to generate a report")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>
            <strong>PriceVision</strong> - AI Stock Trend Predictor<br>
            Built with Streamlit, scikit-learn, and yfinance<br>
            <em>For educational purposes only. Not financial advice.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
