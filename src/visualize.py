"""
Visualization Module
Creates charts and plots for the stock trend predictor.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, List, Any
from matplotlib.figure import Figure


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_price_history(
    df: pd.DataFrame,
    ticker: str,
    show_ma: bool = True,
    interactive: bool = False
) -> Any:
    """
    Plot historical price chart with optional moving averages.

    Args:
        df: DataFrame with Date and Close columns
        ticker: Stock ticker symbol
        show_ma: Whether to show moving averages
        interactive: If True, return plotly figure; else matplotlib

    Returns:
        Plotly or matplotlib figure
    """
    if interactive:
        # Plotly interactive chart
        fig = go.Figure()

        # Add close price
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))

        # Add moving averages if available and requested
        if show_ma:
            if 'ma_5' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['ma_5'],
                    mode='lines',
                    name='5-day MA',
                    line=dict(color='orange', width=1, dash='dash')
                ))
            if 'ma_20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['Date'],
                    y=df['ma_20'],
                    mode='lines',
                    name='20-day MA',
                    line=dict(color='green', width=1, dash='dash')
                ))

        fig.update_layout(
            title=f'{ticker} Stock Price History',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            template='plotly_white',
            height=500
        )

        return fig

    else:
        # Matplotlib static chart
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(df['Date'], df['Close'], label='Close Price', linewidth=2)

        if show_ma:
            if 'ma_5' in df.columns:
                ax.plot(df['Date'], df['ma_5'], label='5-day MA', linestyle='--', alpha=0.7)
            if 'ma_20' in df.columns:
                ax.plot(df['Date'], df['ma_20'], label='20-day MA', linestyle='--', alpha=0.7)

        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.set_title(f'{ticker} Stock Price History', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False
) -> Figure:
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix array
        class_names: List of class names (default: ['DOWN', 'FLAT', 'UP'])
        normalize: Whether to normalize values

    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ['DOWN', 'FLAT', 'UP']

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def plot_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15
) -> Figure:
    """
    Plot feature importance as horizontal bar chart.

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to display

    Returns:
        Matplotlib figure
    """
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute")

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Take top N
    importance_df = importance_df.head(top_n)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))

    ax.barh(
        importance_df['feature'],
        importance_df['importance'],
        color=colors
    )

    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Highest importance at top
    plt.tight_layout()

    return fig


def plot_confidence_bars(confidence: dict) -> Figure:
    """
    Plot prediction confidence as a bar chart.

    Args:
        confidence: Dictionary with trend: probability mappings

    Returns:
        Matplotlib figure
    """
    trends = list(confidence.keys())
    probs = list(confidence.values())

    # Color mapping
    color_map = {
        'UP': '#2ecc71',
        'FLAT': '#95a5a6',
        'DOWN': '#e74c3c'
    }
    colors = [color_map.get(t, '#3498db') for t in trends]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(trends, probs, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{height:.1%}',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

    ax.set_ylabel('Probability', fontsize=12)
    ax.set_xlabel('Trend', fontsize=12)
    ax.set_title('Prediction Confidence', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    return fig


def plot_prediction_history(
    df: pd.DataFrame,
    predictions: pd.Series,
    actuals: Optional[pd.Series] = None
) -> Figure:
    """
    Plot predicted vs actual trends over time.

    Args:
        df: DataFrame with Date and Close columns
        predictions: Series of predicted trends
        actuals: Optional series of actual trends

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot price
    ax1.plot(df['Date'], df['Close'], label='Close Price', color='blue', linewidth=1.5)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title('Stock Price with Predictions', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot predictions
    pred_numeric = predictions.map({'UP': 1, 'FLAT': 0, 'DOWN': -1})
    ax2.plot(df['Date'], pred_numeric, label='Predictions', marker='o', linestyle='-', markersize=3)

    if actuals is not None:
        actual_numeric = actuals.map({'UP': 1, 'FLAT': 0, 'DOWN': -1})
        ax2.plot(df['Date'], actual_numeric, label='Actuals', marker='x', linestyle='--', markersize=3, alpha=0.7)

    ax2.set_ylabel('Trend', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['DOWN', 'FLAT', 'UP'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.5)

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_metrics_summary(metrics: dict) -> Figure:
    """
    Plot summary of model metrics (precision, recall, F1).

    Args:
        metrics: Dictionary with precision, recall, f1_score

    Returns:
        Matplotlib figure
    """
    classes = ['DOWN', 'FLAT', 'UP']

    precision = [metrics['precision'][c] for c in classes]
    recall = [metrics['recall'][c] for c in classes]
    f1 = [metrics['f1_score'][c] for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#e74c3c')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_title('Model Performance Metrics by Class', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom',
                fontsize=9
            )

    plt.tight_layout()

    return fig


def plot_volume_analysis(df: pd.DataFrame) -> Figure:
    """
    Plot volume analysis chart.

    Args:
        df: DataFrame with Date and Volume columns

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    colors = ['green' if df['Close'].iloc[i] >= df['Close'].iloc[i - 1] else 'red'
              for i in range(1, len(df))]
    colors.insert(0, 'gray')

    ax.bar(df['Date'], df['Volume'], color=colors, alpha=0.7)

    ax.set_ylabel('Volume', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Trading Volume', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


if __name__ == "__main__":
    # Quick test
    print("Visualization module loaded successfully")
    print("Available functions:")
    print("- plot_price_history")
    print("- plot_confusion_matrix")
    print("- plot_feature_importance")
    print("- plot_confidence_bars")
    print("- plot_prediction_history")
    print("- plot_metrics_summary")
    print("- plot_volume_analysis")
