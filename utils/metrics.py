"""
Plotly Chart Generators and Data Visualization Helpers.
Produces modern, publication-grade dark-themed interactive charts.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from utils.config import EMOTIONS, EMOTION_META

# Consistent Theme Palette
THEME_BG = "rgba(0,0,0,0)"
THEME_PAPER = "rgba(0,0,0,0)"
FONT_COLOR = "#F8FAFC"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"

def plot_emotion_probabilities(probs: Dict[str, float]) -> go.Figure:
    """Horizontal bar chart showing the confidence distribution across all 5 emotions."""
    categories = list(probs.keys())
    scores = [probs[cat] * 100 for cat in categories]
    colors = [EMOTION_META[cat]["color"] for cat in categories]

    fig = go.Figure(go.Bar(
        x=scores,
        y=categories,
        orientation='h',
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.2)", width=1)),
        text=[f"{s:.1f}%" for s in scores],
        textposition='outside',
        textfont=dict(color=FONT_COLOR, size=12, family="Plus Jakarta Sans")
    ))

    fig.update_layout(
        plot_bgcolor=THEME_BG,
        paper_bgcolor=THEME_PAPER,
        font=dict(color=FONT_COLOR, family="Plus Jakarta Sans"),
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(
            title="Model Probability (%)",
            range=[0, 105],
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False
        ),
        yaxis=dict(
            autorange="reversed",
            showgrid=False
        ),
        height=220
    )
    return fig

def plot_emotion_distribution_pie(df: pd.DataFrame) -> go.Figure:
    """Pie/Donut chart of detected emotions from historical interactions."""
    if df.empty or "primary_emotion" not in df.columns:
        return go.Figure()

    counts = df["primary_emotion"].value_counts().reset_index()
    counts.columns = ["emotion", "count"]
    
    color_map = {emo: EMOTION_META[emo]["color"] for emo in EMOTIONS if emo in counts["emotion"].values}

    fig = px.pie(
        counts,
        names="emotion",
        values="count",
        hole=0.55,
        color="emotion",
        color_discrete_map=color_map
    )

    fig.update_traces(
        textinfo="percent+label",
        hoverinfo="label+value+percent",
        marker=dict(line=dict(color="#0F172A", width=2))
    )

    fig.update_layout(
        plot_bgcolor=THEME_BG,
        paper_bgcolor=THEME_PAPER,
        font=dict(color=FONT_COLOR, family="Plus Jakarta Sans"),
        margin=dict(l=10, r=10, t=20, b=20),
        showlegend=False,
        height=320
    )
    return fig

def plot_emotion_frequency_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart of emotion frequencies."""
    if df.empty or "primary_emotion" not in df.columns:
        return go.Figure()

    counts = df["primary_emotion"].value_counts().reindex(EMOTIONS, fill_value=0).reset_index()
    counts.columns = ["emotion", "count"]
    colors = [EMOTION_META[e]["color"] for e in counts["emotion"]]

    fig = go.Figure(go.Bar(
        x=counts["emotion"],
        y=counts["count"],
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.15)", width=1)),
        text=counts["count"],
        textposition='outside',
        textfont=dict(color=FONT_COLOR, size=13)
    ))

    fig.update_layout(
        plot_bgcolor=THEME_BG,
        paper_bgcolor=THEME_PAPER,
        font=dict(color=FONT_COLOR, family="Plus Jakarta Sans"),
        margin=dict(l=10, r=10, t=20, b=20),
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, title="Interactions"),
        height=320
    )
    return fig

def plot_emotion_trends_line(df: pd.DataFrame) -> go.Figure:
    """Timeline trend of emotions across dates."""
    if df.empty or "timestamp" not in df.columns:
        return go.Figure()

    df_copy = df.copy()
    try:
        df_copy["date"] = pd.to_datetime(df_copy["timestamp"]).dt.date
    except Exception:
        df_copy["date"] = "Today"

    trend = df_copy.groupby(["date", "primary_emotion"]).size().unstack(fill_value=0)
    
    fig = go.Figure()
    for emo in EMOTIONS:
        if emo in trend.columns:
            fig.add_trace(go.Scatter(
                x=[str(d) for d in trend.index],
                y=trend[emo],
                mode='lines+markers',
                name=emo,
                line=dict(color=EMOTION_META[emo]["color"], width=3),
                marker=dict(size=7)
            ))

    fig.update_layout(
        plot_bgcolor=THEME_BG,
        paper_bgcolor=THEME_PAPER,
        font=dict(color=FONT_COLOR, family="Plus Jakarta Sans"),
        margin=dict(l=10, r=10, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR, title="Date"),
        yaxis=dict(showgrid=True, gridcolor=GRID_COLOR, title="Interaction Count"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320
    )
    return fig

def plot_confusion_matrix_heatmap(cm: List[List[int]], classes: List[str], title: str = "Confusion Matrix") -> go.Figure:
    """Interactive heatmap for confusion matrices."""
    cm_arr = np.array(cm)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm_arr,
        x=classes,
        y=classes,
        colorscale=[[0, "#0F172A"], [0.5, "#312E81"], [1.0, "#38BDF8"]],
        text=cm_arr,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "#FFFFFF"},
        hoverongaps=False
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#F8FAFC")),
        plot_bgcolor=THEME_BG,
        paper_bgcolor=THEME_PAPER,
        font=dict(color=FONT_COLOR, family="Plus Jakarta Sans"),
        xaxis=dict(title="Predicted Emotion", showgrid=False),
        yaxis=dict(title="True Emotion", autorange="reversed", showgrid=False),
        margin=dict(l=20, r=20, t=50, b=20),
        height=340
    )
    return fig

def plot_model_comparison_bars(metrics_data: Dict[str, Any]) -> go.Figure:
    """Grouped bar chart comparing BiLSTM and BERT metrics."""
    bilstm = metrics_data.get("bilstm", {})
    bert = metrics_data.get("bert", {})

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    bilstm_vals = [
        bilstm.get("accuracy", 0) * 100,
        bilstm.get("precision", 0) * 100,
        bilstm.get("recall", 0) * 100,
        bilstm.get("f1_score", 0) * 100
    ]
    bert_vals = [
        bert.get("accuracy", 0) * 100,
        bert.get("precision", 0) * 100,
        bert.get("recall", 0) * 100,
        bert.get("f1_score", 0) * 100
    ]

    fig = go.Figure(data=[
        go.Bar(
            name='BiLSTM (PyTorch)',
            x=metrics,
            y=bilstm_vals,
            marker_color='#38BDF8',
            text=[f"{v:.1f}%" for v in bilstm_vals],
            textposition='outside'
        ),
        go.Bar(
            name='BERT (Transformer)',
            x=metrics,
            y=bert_vals,
            marker_color='#818CF8',
            text=[f"{v:.1f}%" for v in bert_vals],
            textposition='outside'
        )
    ])

    fig.update_layout(
        barmode='group',
        plot_bgcolor=THEME_BG,
        paper_bgcolor=THEME_PAPER,
        font=dict(color=FONT_COLOR, family="Plus Jakarta Sans"),
        yaxis=dict(range=[0, 110], showgrid=True, gridcolor=GRID_COLOR, title="Score (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=20),
        height=340
    )
    return fig
