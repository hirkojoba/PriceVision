"""
PDF Report Generation Module
Creates downloadable PDF reports with predictions and metrics.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image as RLImage
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from typing import Dict, Optional, Any
from datetime import datetime
import os
import tempfile


def generate_pdf_report(
    ticker: str,
    start_date: str,
    end_date: str,
    metrics: Dict[str, Any],
    prediction: Dict[str, Any],
    output_path: str,
    chart_paths: Optional[Dict[str, str]] = None
) -> None:
    """
    Generate a comprehensive PDF report.

    Args:
        ticker: Stock ticker symbol
        start_date: Training data start date
        end_date: Training data end date
        metrics: Dictionary with model performance metrics
        prediction: Dictionary with prediction results
        output_path: Path to save the PDF
        chart_paths: Optional dict with paths to chart images
    """
    # Create document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )

    # Container for the 'Flowable' objects
    elements = []

    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )

    normal_style = styles['Normal']
    normal_style.fontSize = 11
    normal_style.leading = 14

    # Title
    title = Paragraph("AI Stock Trend Predictor Report", title_style)
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Subtitle
    subtitle = Paragraph(f"<b>PriceVision</b> - Powered by Machine Learning", styles['Normal'])
    subtitle.alignment = TA_CENTER
    elements.append(subtitle)
    elements.append(Spacer(1, 20))

    # Report metadata
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta_data = [
        ['Report Generated:', report_date],
        ['Stock Ticker:', ticker],
        ['Training Period:', f'{start_date} to {end_date}'],
    ]

    meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))

    elements.append(meta_table)
    elements.append(Spacer(1, 20))

    # Prediction Section
    elements.append(Paragraph("Next-Day Trend Prediction", heading_style))

    trend = prediction.get('trend', 'N/A')
    confidence = prediction.get('confidence', {})

    # Trend color coding
    trend_color = {
        'UP': colors.green,
        'DOWN': colors.red,
        'FLAT': colors.grey
    }.get(trend, colors.black)

    pred_data = [
        ['Predicted Trend:', f'<font color="{trend_color.hexval()}"><b>{trend}</b></font>'],
        ['Latest Close Price:', f"${prediction.get('latest_close', 0):.2f}"],
        ['Latest Date:', prediction.get('latest_date', 'N/A')],
        ['Prediction Date:', prediction.get('next_day_prediction_date', 'N/A')],
    ]

    pred_table = Table(pred_data, colWidths=[2*inch, 4*inch])
    pred_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))

    elements.append(pred_table)
    elements.append(Spacer(1, 15))

    # Confidence scores
    elements.append(Paragraph("Prediction Confidence", heading_style))

    conf_data = [['Trend', 'Probability']]
    for trend_name, prob in confidence.items():
        conf_data.append([trend_name, f'{prob:.2%}'])

    conf_table = Table(conf_data, colWidths=[2*inch, 2*inch])
    conf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
    ]))

    elements.append(conf_table)
    elements.append(Spacer(1, 20))

    # Model Performance Section
    elements.append(Paragraph("Model Performance Metrics", heading_style))

    accuracy = metrics.get('accuracy', 0)
    perf_data = [
        ['Overall Accuracy:', f'{accuracy:.2%}'],
    ]

    # Add per-class metrics
    for class_name in ['UP', 'DOWN', 'FLAT']:
        precision = metrics.get('precision', {}).get(class_name, 0)
        recall = metrics.get('recall', {}).get(class_name, 0)
        f1 = metrics.get('f1_score', {}).get(class_name, 0)

        perf_data.append([f'{class_name} - Precision:', f'{precision:.2%}'])
        perf_data.append([f'{class_name} - Recall:', f'{recall:.2%}'])
        perf_data.append([f'{class_name} - F1-Score:', f'{f1:.2%}'])

    perf_table = Table(perf_data, colWidths=[2.5*inch, 1.5*inch])
    perf_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
    ]))

    elements.append(perf_table)
    elements.append(Spacer(1, 20))

    # Add charts if provided
    if chart_paths:
        for chart_name, chart_path in chart_paths.items():
            if os.path.exists(chart_path):
                elements.append(PageBreak())
                elements.append(Paragraph(f"{chart_name}", heading_style))
                try:
                    img = RLImage(chart_path, width=6*inch, height=4*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 12))
                except Exception as e:
                    elements.append(Paragraph(f"Error loading chart: {str(e)}", normal_style))

    # Disclaimer
    elements.append(Spacer(1, 30))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#7f8c8d'),
        leading=12,
        borderColor=colors.HexColor('#e74c3c'),
        borderWidth=2,
        borderPadding=10,
        backColor=colors.HexColor('#fff5f5')
    )

    disclaimer_text = """
    <b>DISCLAIMER:</b> This report is generated for <b>educational purposes only</b> and does not constitute
    financial advice. The predictions are based on historical data and machine learning models, which
    may not accurately predict future stock movements. Do not use this information for actual trading
    or investment decisions. Always consult with a qualified financial advisor before making any
    investment decisions. Past performance does not guarantee future results.
    """

    disclaimer = Paragraph(disclaimer_text, disclaimer_style)
    elements.append(disclaimer)

    # Footer
    elements.append(Spacer(1, 20))
    footer_text = f"<i>Generated by PriceVision AI Stock Trend Predictor | {report_date}</i>"
    footer = Paragraph(footer_text, styles['Normal'])
    footer.alignment = TA_CENTER
    elements.append(footer)

    # Build PDF
    doc.build(elements)


def create_simple_report(
    ticker: str,
    prediction: Dict[str, Any],
    output_path: str
) -> None:
    """
    Create a simplified single-page report.

    Args:
        ticker: Stock ticker
        prediction: Prediction results
        output_path: Output file path
    """
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Title
    title = Paragraph(f"Stock Trend Prediction: {ticker}", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 20))

    # Prediction
    trend = prediction.get('trend', 'N/A')
    text = f"""
    <b>Predicted Trend:</b> {trend}<br/>
    <b>Confidence:</b><br/>
    UP: {prediction.get('confidence', {}).get('UP', 0):.1%}<br/>
    FLAT: {prediction.get('confidence', {}).get('FLAT', 0):.1%}<br/>
    DOWN: {prediction.get('confidence', {}).get('DOWN', 0):.1%}<br/>
    """
    elements.append(Paragraph(text, styles['Normal']))
    elements.append(Spacer(1, 20))

    # Disclaimer
    disclaimer = Paragraph(
        "<b>DISCLAIMER:</b> For educational purposes only. Not financial advice.",
        styles['Normal']
    )
    elements.append(disclaimer)

    doc.build(elements)


if __name__ == "__main__":
    # Quick test
    print("Testing PDF report generation...")

    test_metrics = {
        'accuracy': 0.65,
        'precision': {'UP': 0.70, 'DOWN': 0.60, 'FLAT': 0.65},
        'recall': {'UP': 0.68, 'DOWN': 0.62, 'FLAT': 0.60},
        'f1_score': {'UP': 0.69, 'DOWN': 0.61, 'FLAT': 0.62}
    }

    test_prediction = {
        'trend': 'UP',
        'confidence': {'UP': 0.62, 'FLAT': 0.20, 'DOWN': 0.18},
        'latest_close': 182.34,
        'latest_date': '2024-01-15',
        'next_day_prediction_date': '2024-01-16'
    }

    try:
        output_file = "/tmp/test_report.pdf"
        generate_pdf_report(
            ticker='AAPL',
            start_date='2023-01-01',
            end_date='2024-01-01',
            metrics=test_metrics,
            prediction=test_prediction,
            output_path=output_file
        )
        print(f"Test report generated successfully: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
