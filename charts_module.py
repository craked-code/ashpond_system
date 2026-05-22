import plotly.graph_objects as go
import numpy as np
import config

# ── Colour palette ────────────────────────────────────────────────────────────
COLOR_MAP = {
    "CRITICAL": "#ef4444",
    "ELEVATED": "#f59e0b",
    "STABLE":   "#22c55e",
}

# ── Shared Plotly dark layout ─────────────────────────────────────────────────
PLOTLY_DARK_LAYOUT = dict(
    plot_bgcolor="#0d0f1a",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="'JetBrains Mono', 'Courier New', monospace",
        color="#64748b",
        size=12,
    ),
    title=dict(
        font=dict(
            family="'Inter', sans-serif",
            color="#e2e8f0",
            size=14,
            weight="bold",
        )
    ),
    xaxis=dict(
        gridcolor="rgba(0,240,255,0.06)",
        zerolinecolor="rgba(0,240,255,0.12)",
        color="#64748b",
        linecolor="rgba(0,240,255,0.1)",
    ),
    yaxis=dict(
        gridcolor="rgba(0,240,255,0.06)",
        zerolinecolor="rgba(0,240,255,0.12)",
        color="#64748b",
        linecolor="rgba(0,240,255,0.1)",
    ),
    hoverlabel=dict(
        bgcolor="#0d1120",
        bordercolor="rgba(0,240,255,0.3)",
        font=dict(
            family="'JetBrains Mono', monospace",
            color="#e2e8f0",
            size=12,
        ),
    ),
    margin=dict(l=10, r=80, t=60, b=40),
)


def _merge_layout(overrides: dict) -> dict:
    """Deep-merge overrides into PLOTLY_DARK_LAYOUT without mutating the base."""
    import copy
    base = copy.deepcopy(PLOTLY_DARK_LAYOUT)
    base.update(overrides)
    return base


# ── DII Bar Chart ─────────────────────────────────────────────────────────────
def build_dii_bar_chart(df):
    df_s   = df.sort_values('dii_score', ascending=True).copy()
    colors = [COLOR_MAP[c] for c in df_s['risk_category']]
    labels = [
        f"{n}  {t}"
        for n, t in zip(df_s['pond_name'],
                        df_s.get('dii_trend', ['→'] * len(df_s)))
    ]
    lowers = df_s.get('dii_lower', df_s['dii_score']).values
    uppers = df_s.get('dii_upper', df_s['dii_score']).values
    errors = uppers - df_s['dii_score'].values

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_s['dii_score'],
        y=labels,
        orientation='h',
        marker=dict(
            color=colors,
            opacity=0.88,
            line=dict(color="rgba(0,240,255,0.15)", width=1),
        ),
        text=[f"{s:.2f}" for s in df_s['dii_score']],
        textposition='outside',
        textfont=dict(family="'JetBrains Mono', monospace", color="#94a3b8", size=11),
        error_x=dict(
            type='data',
            array=errors,
            arrayminus=df_s['dii_score'].values - lowers,
            visible=True,
            color='rgba(0,240,255,0.3)',
            thickness=2,
        ),
        hovertemplate="<b>%{y}</b><br>DII: %{x:.3f}<extra></extra>",
    ))

    # Risk zone background bands
    fig.add_vrect(x0=0,                        x1=config.ELEVATED_THRESHOLD,
                  fillcolor="rgba(34,197,94,0.04)",  layer="below", line_width=0)
    fig.add_vrect(x0=config.ELEVATED_THRESHOLD, x1=config.CRITICAL_THRESHOLD,
                  fillcolor="rgba(245,158,11,0.04)", layer="below", line_width=0)
    fig.add_vrect(x0=config.CRITICAL_THRESHOLD, x1=1.15,
                  fillcolor="rgba(239,68,68,0.04)",  layer="below", line_width=0)

    # Threshold lines
    fig.add_vline(x=config.CRITICAL_THRESHOLD, line_dash="dash",
                  line_color="#ef4444", line_width=2,
                  annotation_text="CRITICAL (0.70)",
                  annotation_font_color="#ef4444",
                  annotation_font_size=11)
    fig.add_vline(x=config.ELEVATED_THRESHOLD, line_dash="dot",
                  line_color="#f59e0b", line_width=1.5,
                  annotation_text="ELEVATED (0.40)",
                  annotation_position="bottom right",
                  annotation_font_color="#f59e0b",
                  annotation_font_size=11)

    layout = _merge_layout(dict(
        title="Dyke Instability Index — All Ponds (error bars = 95% CI)",
        height=500,
        showlegend=False,
        xaxis=dict(
            **PLOTLY_DARK_LAYOUT['xaxis'],
            range=[0, 1.15],
            title="DII Score",
            tickformat=".2f",
        ),
        margin=dict(l=10, r=80, t=60, b=40),
    ))
    fig.update_layout(**layout)
    return fig


# ── SHAP Waterfall ────────────────────────────────────────────────────────────
def build_shap_waterfall(shap_dict, pond_name):
    features = ['NDWI (Seepage)', 'NDVI Stress', 'Slope Risk', 'Rainfall', 'Proximity']
    values   = [
        shap_dict['ndwi'], shap_dict['ndvi'], shap_dict['slope'],
        shap_dict['rainfall'], shap_dict['proximity'],
    ]
    base   = shap_dict['base_value']
    colors = ['#ef4444' if v > 0 else '#22c55e' for v in values]
    texts  = [f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=features,
        orientation='h',
        marker=dict(color=colors, opacity=0.88,
                    line=dict(color="rgba(0,240,255,0.1)", width=1)),
        text=texts,
        textposition='outside',
        textfont=dict(family="'JetBrains Mono', monospace", color="#94a3b8", size=11),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))

    fig.add_vline(x=0, line_color='rgba(0,240,255,0.3)', line_width=1.5)
    fig.add_vline(x=base, line_dash='dot', line_color='#7c3aed', line_width=1.5,
                  annotation_text=f"base={base:.3f}",
                  annotation_font_color='#a78bfa',
                  annotation_font_size=10)

    layout = _merge_layout(dict(
        title=f"SHAP Breach Risk Explanation — {pond_name}",
        height=320,
        showlegend=False,
        xaxis=dict(
            **PLOTLY_DARK_LAYOUT['xaxis'],
            title="Contribution to breach probability",
            tickformat=".3f",
        ),
        margin=dict(l=10, r=60, t=50, b=40),
    ))
    fig.update_layout(**layout)
    return fig


# ── Factor Breakdown ──────────────────────────────────────────────────────────
def build_factor_breakdown_chart(row):
    factors = {
        "NDWI (Seepage)":    min(row['ndwi_score'] / config.NDWI_MAX, 1.0),
        "NDVI Stress":       min((1 - row['ndvi_score']) / config.NDVI_STRESS_MAX, 1.0),
        "Slope Risk":        min(row['slope_degrees'] / config.SLOPE_MAX, 1.0),
        "Rainfall Exposure": min(row['rainfall_forecast_mm'] / config.RAINFALL_MAX, 1.0),
        "SAR Backscatter":   min(row.get('sar_backscatter', 0) / config.SAR_MAX, 1.0),
        "Historical Prox.":  row['breach_proximity_score'],
    }
    names  = list(factors.keys())
    values = list(factors.values())
    colors = [
        '#ef4444' if v > 0.70 else '#f59e0b' if v > 0.40 else '#22c55e'
        for v in values
    ]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation='h',
        marker=dict(color=colors, opacity=0.88,
                    line=dict(color="rgba(0,240,255,0.1)", width=1)),
        text=[f"{v*100:.0f}%" for v in values],
        textposition='outside',
        textfont=dict(family="'JetBrains Mono', monospace", color="#94a3b8", size=11),
        hovertemplate="<b>%{y}</b><br>%{x:.2f}<extra></extra>",
    ))

    layout = _merge_layout(dict(
        title=f"Factor Breakdown — {row['pond_name']}",
        height=340,
        showlegend=False,
        xaxis=dict(
            **PLOTLY_DARK_LAYOUT['xaxis'],
            range=[0, 1.2],
        ),
        margin=dict(l=10, r=60, t=50, b=40),
    ))
    fig.update_layout(**layout)
    return fig


# ── Retrospective Timeline ────────────────────────────────────────────────────
def build_retrospective_chart(retro_df):
    import ruptures as rpt

    signal = retro_df['dii_score'].values
    point_colors = [
        '#22c55e' if s < 0.40 else '#f59e0b' if s < 0.70 else '#ef4444'
        for s in signal
    ]

    fig = go.Figure()

    # Risk zone horizontal bands
    fig.add_hrect(y0=0,                        y1=config.ELEVATED_THRESHOLD,
                  fillcolor="rgba(34,197,94,0.03)",  layer="below", line_width=0)
    fig.add_hrect(y0=config.ELEVATED_THRESHOLD, y1=config.CRITICAL_THRESHOLD,
                  fillcolor="rgba(245,158,11,0.03)", layer="below", line_width=0)
    fig.add_hrect(y0=config.CRITICAL_THRESHOLD, y1=1.05,
                  fillcolor="rgba(239,68,68,0.03)",  layer="below", line_width=0)

    # Main DII line with cyan + fill
    fig.add_trace(go.Scatter(
        x=retro_df['date'], y=signal,
        mode='lines+markers',
        line=dict(color='#00f0ff', width=2.5),
        marker=dict(size=10, color=point_colors,
                    line=dict(color='rgba(0,240,255,0.3)', width=1)),
        fill='tozeroy',
        fillcolor='rgba(0,240,255,0.04)',
        name='Simulated DII',
        hovertemplate="<b>%{x|%d %b %Y}</b><br>DII: %{y:.3f}<extra></extra>",
    ))

    # Change point detection
    try:
        model_cpd  = rpt.Pelt(model="rbf").fit(signal)
        breakpoints = model_cpd.predict(pen=0.8)
        for bp in breakpoints[:-1]:
            bp_date = retro_df['date'].iloc[min(bp, len(retro_df) - 1)]
            fig.add_vline(
                x=str(bp_date)[:10],
                line_dash="dot", line_color="#a78bfa", line_width=2,
                annotation_text="BEHAVIOUR CHANGE DETECTED",
                annotation_position="top left",
                annotation_font_color="#a78bfa",
                annotation_font_size=10,
            )
    except Exception:
        pass

    # Critical threshold line
    fig.add_hline(y=0.70, line_dash="dash", line_color="#ef4444", line_width=2,
                  annotation_text="CRITICAL THRESHOLD (0.70)",
                  annotation_position="right",
                  annotation_font_color="#ef4444",
                  annotation_font_size=10)

    # Elevated threshold line
    fig.add_hline(y=0.40, line_dash="dot", line_color="#f59e0b", line_width=1,
                  annotation_text="ELEVATED (0.40)",
                  annotation_position="right",
                  annotation_font_color="#f59e0b",
                  annotation_font_size=10)

    # Breach event marker
    breach = retro_df[retro_df['event'] == 'BREACH']
    if not breach.empty:
        fig.add_trace(go.Scatter(
            x=breach['date'], y=breach['dii_score'],
            mode='markers', name='ACTUAL BREACH Apr 20 2020',
            marker=dict(size=18, color='#ef4444', symbol='x',
                        line=dict(width=3, color='#ef4444')),
        ))

    # 13-day early warning window
    fig.add_vrect(
        x0="2020-04-07", x1="2020-04-20",
        fillcolor="rgba(239,68,68,0.08)", opacity=1, line_width=0,
        annotation_text="13-DAY EARLY WARNING WINDOW",
        annotation_position="top left",
        annotation_font_color="#fca5a5",
        annotation_font_size=10,
    )

    layout = _merge_layout(dict(
        title="Retrospective Simulation — Sasan UMPP 2020 · Change Point Detection",
        height=480,
        xaxis=dict(**PLOTLY_DARK_LAYOUT['xaxis'], title="Date"),
        yaxis=dict(**PLOTLY_DARK_LAYOUT['yaxis'], range=[0, 1.05], tickformat=".2f"),
        margin=dict(l=10, r=20, t=60, b=130),
        legend=dict(
            orientation='h', yanchor='top', y=-0.18,
            font=dict(color="#64748b", family="'Inter', sans-serif"),
            bgcolor="rgba(0,0,0,0)",
        ),
    ))
    fig.update_layout(**layout)

    fig.add_annotation(
        text="⚠ " + config.RETRO_DISCLAIMER,
        xref="paper", yref="paper",
        x=0, y=-0.34, showarrow=False,
        font=dict(size=10, color="#475569", family="'Inter', sans-serif"),
        align="left",
    )
    return fig