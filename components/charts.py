"""Plotly analytical chart components with unified green-and-white AgriGuard styling."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from config.settings import CLASS_NAMES

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFFFF",
    font=dict(family="-apple-system, BlinkMacSystemFont, Inter, sans-serif", color="#142419", size=12),
    margin=dict(l=46, r=36, t=54, b=82),
    hovermode="x unified",
)


def _apply_clean_axes(fig: go.Figure, yaxis_title: str = "", xaxis_title: str = "") -> go.Figure:
    fig.update_xaxes(showgrid=True, gridcolor="#EDF2EE", linecolor="#DCE6DE", title_text=xaxis_title)
    fig.update_yaxes(showgrid=True, gridcolor="#EDF2EE", linecolor="#DCE6DE")
    fig.update_layout(**_LAYOUT_DEFAULTS)
    if yaxis_title:
        fig.update_layout(yaxis_title=yaxis_title)
    return fig


def plot_seven_day_risk_chart(days: list[dict[str, Any]]) -> go.Figure:
    """Create 7-day Environmental Disease-Risk Index chart with threshold bands."""
    dates = [d["date"] for d in days]
    scores = [d["score"] for d in days]
    colors = [d["color"] for d in days]
    cats = [d["category"] for d in days]

    fig = go.Figure()
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(46, 125, 50, 0.07)", line_width=0)
    fig.add_hrect(y0=30, y1=60, fillcolor="rgba(217, 119, 6, 0.07)", line_width=0)
    fig.add_hrect(y0=60, y1=80, fillcolor="rgba(220, 38, 38, 0.07)", line_width=0)
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(127, 29, 29, 0.09)", line_width=0)

    fig.add_trace(
        go.Bar(
            x=dates,
            y=scores,
            marker_color=colors,
            name="Risk Index",
            customdata=cats,
            hovertemplate="<b>%{x}</b><br>Score: %{y:.1f}/100 (%{customdata})<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=scores,
            mode="lines+markers",
            line=dict(color="#134417", width=2.2),
            marker=dict(size=7, color="#1B5E20"),
            name="7-Day Trend",
            hoverinfo="skip",
        )
    )
    fig.update_yaxes(range=[0, 105])
    fig.update_layout(
        title=dict(text="7-Day Environmental Disease-Risk Index (0–100)", x=0.0, xanchor="left"),
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5),
    )
    return _apply_clean_axes(fig, yaxis_title="Risk Index (0–100)")


def plot_risk_factor_contributions(days: list[dict[str, Any]]) -> go.Figure:
    """Stacked bar chart decomposing daily risk score by contributing weather factor."""
    dates = [d["date"] for d in days]
    factor_keys = [
        ("Relative Humidity (30%)", "#1B5E20"),
        ("Temperature Suitability (25%)", "#4CAF50"),
        ("Rainfall Splash (25%)", "#0284C7"),
        ("Dew / Wetness Proxy (20%)", "#D97706"),
    ]
    fig = go.Figure()
    for key, color in factor_keys:
        vals = [d.get("contributions", {}).get(key, 0.0) for d in days]
        fig.add_trace(go.Bar(x=dates, y=vals, name=key, marker_color=color))

    fig.update_layout(
        barmode="stack",
        title=dict(text="Daily Weighted Factor Contributions to Risk Index", x=0.0, xanchor="left"),
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5),
    )
    fig.update_yaxes(range=[0, 105])
    return _apply_clean_axes(fig, yaxis_title="Weighted Points")


def plot_temperature_trend(days: list[dict[str, Any]]) -> go.Figure:
    """Plot 7-day min/mean/max temperature against the 24–29°C Alternaria optimum zone."""
    dates = [d["date"] for d in days]
    t_max = [d.get("temp_max") for d in days]
    t_mean = [d.get("temp_mean") for d in days]
    t_min = [d.get("temp_min") for d in days]

    fig = go.Figure()
    fig.add_hrect(
        y0=24,
        y1=29,
        fillcolor="rgba(217, 119, 6, 0.12)",
        line_width=0,
        annotation_text="Optimal A. solani range (24–29°C)",
        annotation_position="top left",
    )
    fig.add_trace(go.Scatter(x=dates, y=t_max, name="Max °C", line=dict(color="#DC2626", width=1.8, dash="dot")))
    fig.add_trace(go.Scatter(x=dates, y=t_mean, name="Mean °C", mode="lines+markers", line=dict(color="#1B5E20", width=2.8)))
    fig.add_trace(go.Scatter(x=dates, y=t_min, name="Min °C", line=dict(color="#0284C7", width=1.8, dash="dot")))
    fig.update_layout(
        title=dict(text="7-Day Temperature Profile (°C)", x=0.0, xanchor="left"),
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5),
    )
    return _apply_clean_axes(fig, yaxis_title="Temperature (°C)")


def plot_humidity_dew_trend(days: list[dict[str, Any]]) -> go.Figure:
    """Plot daily mean & max relative humidity plus dew-point depression."""
    dates = [d["date"] for d in days]
    rh_mean = [d.get("rh_mean") for d in days]
    rh_max = [d.get("rh_max") for d in days]
    hrs_85 = [d.get("hours_rh_ge_85") for d in days]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=rh_max, name="Daily Peak RH (%)", line=dict(color="#2E7D32", width=1.6, dash="dash")))
    fig.add_trace(go.Scatter(x=dates, y=rh_mean, name="Daily Mean RH (%)", mode="lines+markers", line=dict(color="#1B5E20", width=2.6)))
    if any(v is not None for v in hrs_85):
        fig.add_trace(go.Bar(x=dates, y=hrs_85, name="Hours RH ≥ 85%", marker_color="rgba(76, 175, 80, 0.30)"))
    fig.update_yaxes(range=[0, 105])
    fig.update_layout(
        title=dict(text="Relative Humidity & High-Moisture Duration", x=0.0, xanchor="left"),
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5),
    )
    return _apply_clean_axes(fig, yaxis_title="% RH / Hours ≥ 85%")


def plot_rainfall_chart(days: list[dict[str, Any]]) -> go.Figure:
    """Plot daily precipitation totals (mm) and max rain probability (%)."""
    dates = [d["date"] for d in days]
    rain = [d.get("precipitation_sum") or 0.0 for d in days]
    prob = [d.get("precip_prob_max") for d in days]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=dates, y=rain, name="Precipitation (mm)", marker_color="#0284C7"))
    if any(v is not None for v in prob):
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prob,
                name="Max Rain Prob (%)",
                mode="lines+markers",
                line=dict(color="#1B5E20", width=2),
                yaxis="y2",
            )
        )
    max_rain = max(rain) if rain else 0.0
    fig.update_layout(
        title=dict(text="7-Day Forecast Rainfall & Probability", x=0.0, xanchor="left"),
        yaxis=dict(range=[0, max(max_rain * 1.25, 5.0)], rangemode="nonnegative"),
        yaxis2=dict(title="Probability (%)", overlaying="y", side="right", range=[0, 105], showgrid=False),
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5),
    )
    return _apply_clean_axes(fig, yaxis_title="Rainfall (mm)")


def plot_enso_history_chart(roni_series: list[dict[str, Any]], oni_series: list[dict[str, Any]] | None = None) -> go.Figure:
    """Plot official NOAA RONI (primary) and ONI (reference) 3-month seasonal trajectories."""
    labels = [r["label"] for r in roni_series]
    roni_vals = [r["anomaly"] for r in roni_series]

    fig = go.Figure()
    fig.add_hrect(y0=0.5, y1=3.0, fillcolor="rgba(220, 38, 38, 0.07)", line_width=0)
    fig.add_hrect(y0=-3.0, y1=-0.5, fillcolor="rgba(2, 132, 199, 0.07)", line_width=0)
    fig.add_hline(y=0.5, line_dash="dash", line_color="#DC2626", annotation_text="El Niño (+0.5°C)")
    fig.add_hline(y=-0.5, line_dash="dash", line_color="#0284C7", annotation_text="La Niña (-0.5°C)")
    fig.add_hline(y=0.0, line_color="#94A3B8", line_width=1)

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=roni_vals,
            mode="lines+markers",
            name="NOAA RONI (Primary Operational)",
            line=dict(color="#1B5E20", width=2.8),
            marker=dict(size=6),
        )
    )
    if oni_series:
        oni_map = {r["label"]: r["anomaly"] for r in oni_series}
        aligned_oni = [oni_map.get(lbl) for lbl in labels]
        fig.add_trace(
            go.Scatter(
                x=labels,
                y=aligned_oni,
                mode="lines",
                name="Traditional ONI (Reference)",
                line=dict(color="#64748B", width=1.8, dash="dot"),
            )
        )

    fig.update_layout(
        title=dict(text="NOAA CPC 3-Month Running Mean ENSO Indices (Last 36 Seasons)", x=0.0, xanchor="left"),
        legend=dict(orientation="h", yanchor="top", y=-0.24, xanchor="center", x=0.5),
    )
    fig.update_xaxes(tickangle=-35, nticks=12)
    out = _apply_clean_axes(fig, yaxis_title="SST Anomaly (°C)")
    out.update_layout(margin=dict(l=48, r=24, t=54, b=96))
    return out


def plot_what_if_comparison(baseline: dict[str, Any], hypothetical: dict[str, Any]) -> go.Figure:
    """Compare baseline vs hypothetical scenario contributions."""
    factors = list((baseline.get("contributions") or {}).keys())
    base_vals = [baseline["contributions"].get(k, 0.0) for k in factors]
    hyp_vals = [hypothetical["contributions"].get(k, 0.0) for k in factors]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=factors, y=base_vals, name="Baseline Forecast", marker_color="#64748B"))
    fig.add_trace(go.Bar(x=factors, y=hyp_vals, name="Hypothetical Scenario", marker_color="#1B5E20"))
    fig.update_layout(
        barmode="group",
        title=dict(text="Factor Contribution Comparison (Baseline vs Hypothetical)", x=0.0, xanchor="left"),
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5),
    )
    return _apply_clean_axes(fig, yaxis_title="Weighted Risk Points")


def plot_training_curves_plotly(history_payload: dict[str, Any]) -> go.Figure:
    """Interactive Plotly accuracy/loss curves from actual training logs."""
    hist = history_payload.get("history") or {}
    epochs = list(range(1, len(hist.get("loss", [])) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=hist.get("accuracy", []), name="Train Accuracy", mode="lines+markers", line=dict(color="#1B5E20", width=2.4)))
    fig.add_trace(go.Scatter(x=epochs, y=hist.get("val_accuracy", []), name="Val Accuracy", mode="lines+markers", line=dict(color="#4CAF50", width=2.4)))
    fig.add_trace(go.Scatter(x=epochs, y=hist.get("loss", []), name="Train Loss", mode="lines+markers", line=dict(color="#DC2626", width=2, dash="dot"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=epochs, y=hist.get("val_loss", []), name="Val Loss", mode="lines+markers", line=dict(color="#D97706", width=2, dash="dot"), yaxis="y2"))

    stage1 = history_payload.get("stage1_epochs", 0)
    if 0 < stage1 < len(epochs):
        fig.add_vline(
            x=stage1 + 0.5,
            line_dash="dash",
            line_color="#64748B",
            annotation_text="Fine-tuning start",
            annotation_position="bottom right",
        )

    out = _apply_clean_axes(fig, yaxis_title="Accuracy", xaxis_title="Epoch")
    out.update_layout(
        title=dict(text="Training & Validation Accuracy / Loss Progression", x=0.0, xanchor="left"),
        yaxis=dict(title="Accuracy", range=[0, 1.05]),
        yaxis2=dict(title="Loss", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5),
        margin=dict(l=52, r=56, t=54, b=82),
    )
    return out


def plot_confusion_matrix_plotly(cm: list[list[int]], title: str) -> go.Figure:
    """Interactive confusion matrix heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale=[[0.0, "#F0F7F1"], [1.0, "#1B5E20"]],
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            showscale=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        yaxis_autorange="reversed",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=46, b=40),
    )
    return fig


def plot_prediction_distribution_plotly(
    dist: dict[str, Any],
    title: str = "Ground-Truth vs. Predicted Class Distribution",
) -> go.Figure:
    """Grouped bar chart illustrating ground-truth vs predicted counts across classes."""
    gt = dist.get("ground_truth_counts") or {}
    pred = dist.get("predicted_counts") or {}
    gt_vals = [gt.get(cls, 0) for cls in CLASS_NAMES]
    pred_vals = [pred.get(cls, 0) for cls in CLASS_NAMES]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=CLASS_NAMES, y=gt_vals, name="Ground Truth Count", marker_color="#64748B", text=gt_vals, textposition="outside"))
    fig.add_trace(go.Bar(x=CLASS_NAMES, y=pred_vals, name="Model Predicted Count", marker_color="#D97706", text=pred_vals, textposition="outside"))
    fig.update_layout(
        barmode="group",
        title=dict(text=title, x=0.0, xanchor="left"),
        legend=dict(orientation="h", yanchor="top", y=-0.20, xanchor="center", x=0.5),
    )
    max_val = max(gt_vals + pred_vals + [10])
    fig.update_yaxes(range=[0, max_val * 1.18])
    return _apply_clean_axes(fig, yaxis_title="Image Count")

