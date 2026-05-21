import plotly.graph_objects as go
import numpy as np
import config

COLOR_MAP = {"CRITICAL":"#d32f2f","ELEVATED":"#f57c00","STABLE":"#2e7d32"}

def build_dii_bar_chart(df):
    df_s    = df.sort_values('dii_score', ascending=True).copy()
    colors  = [COLOR_MAP[c] for c in df_s['risk_category']]
    labels  = [f"{n}  {t}" for n,t in zip(df_s['pond_name'], df_s.get('dii_trend', ['→']*len(df_s)))]
    lowers  = df_s.get('dii_lower', df_s['dii_score']).values
    uppers  = df_s.get('dii_upper', df_s['dii_score']).values
    errors  = uppers - df_s['dii_score'].values

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_s['dii_score'],
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f"{s:.2f}" for s in df_s['dii_score']],
        textposition='outside',
        error_x=dict(
            type='data',
            array=errors,
            arrayminus=df_s['dii_score'].values - lowers,
            visible=True,
            color='rgba(0,0,0,0.4)',
            thickness=2,
        ),
    ))

    fig.add_vline(x=config.CRITICAL_THRESHOLD, line_dash="dash",
                  line_color="#d32f2f", line_width=2.5,
                  annotation_text="CRITICAL (0.70)",
                  annotation_font_color="#d32f2f")
    fig.add_vline(x=config.ELEVATED_THRESHOLD, line_dash="dot",
                  line_color="#f57c00", line_width=1.5,
                  annotation_text="ELEVATED (0.40)",
                  annotation_position="bottom right",
                  annotation_font_color="#f57c00")
    fig.update_layout(
        title="Dyke Instability Index — All 10 Ponds (error bars = 95% confidence)",
        xaxis=dict(range=[0,1.15], title="DII Score"),
        height=500,
        margin=dict(l=10,r=80,t=60,b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    return fig

def build_shap_waterfall(shap_dict, pond_name):
    features = ['NDWI (Seepage)', 'NDVI Stress', 'Slope Risk', 'Rainfall', 'Proximity']
    values   = [shap_dict['ndwi'], shap_dict['ndvi'], shap_dict['slope'],
                shap_dict['rainfall'], shap_dict['proximity']]
    base     = shap_dict['base_value']
    colors   = ['#d32f2f' if v > 0 else '#388e3c' for v in values]
    texts    = [f"+{v:.3f}" if v > 0 else f"{v:.3f}" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=features,
        orientation='h',
        marker_color=colors,
        text=texts,
        textposition='outside',
    ))
    fig.add_vline(x=0, line_color='black', line_width=1)
    fig.add_vline(x=base, line_dash='dot', line_color='gray', line_width=1,
                  annotation_text=f"base={base:.3f}",
                  annotation_font_color='gray')
    fig.update_layout(
        title=f"SHAP Breach Risk Explanation — {pond_name}",
        xaxis_title="Contribution to breach probability (red=increases risk)",
        height=320,
        margin=dict(l=10,r=60,t=50,b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    return fig

def build_factor_breakdown_chart(row):
    factors = {
        "NDWI (Seepage)":     min(row['ndwi_score'] / config.NDWI_MAX, 1.0),
        "NDVI Stress":        min((1 - row['ndvi_score']) / config.NDVI_STRESS_MAX, 1.0),
        "Slope Risk":         min(row['slope_degrees'] / config.SLOPE_MAX, 1.0),
        "Rainfall Exposure":  min(row['rainfall_forecast_mm'] / config.RAINFALL_MAX, 1.0),
        "SAR Backscatter":    min(row.get('sar_backscatter', 0) / config.SAR_MAX, 1.0),
        "Historical Prox.":   row['breach_proximity_score'],
    }
    names  = list(factors.keys())
    values = list(factors.values())
    colors = ['#d32f2f' if v > 0.70 else '#f57c00' if v > 0.40 else '#2e7d32' for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=names,
        orientation='h',
        marker_color=colors,
        text=[f"{v*100:.0f}%" for v in values],
        textposition='outside',
    ))
    fig.update_layout(
        title=f"Factor Breakdown — {row['pond_name']}",
        xaxis=dict(range=[0,1.2]),
        height=340,
        margin=dict(l=10,r=60,t=50,b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    return fig

def build_retrospective_chart(retro_df):
    import ruptures as rpt

    signal = retro_df['dii_score'].values
    colors = ['#2e7d32' if s < 0.40 else '#f57c00' if s < 0.70 else '#d32f2f'
              for s in signal]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=retro_df['date'], y=signal,
        mode='lines+markers',
        line=dict(color='#1565c0', width=3),
        marker=dict(size=10, color=colors),
        name='Simulated DII',
    ))

    try:
        model_cpd = rpt.Pelt(model="rbf").fit(signal)
        breakpoints = model_cpd.predict(pen=0.8)
        for bp in breakpoints[:-1]:
            bp_date = retro_df['date'].iloc[min(bp, len(retro_df)-1)]
            fig.add_vline(
                x=str(bp_date)[:10],
                line_dash="dot", line_color="#7b1fa2", line_width=2,
                annotation_text="BEHAVIOUR CHANGE DETECTED",
                annotation_position="top left",
                annotation_font_color="#7b1fa2",
            )
    except Exception:
        pass

    fig.add_hline(y=0.70, line_dash="dash", line_color="#d32f2f", line_width=2,
                  annotation_text="CRITICAL THRESHOLD (0.70)",
                  annotation_position="right",
                  annotation_font_color="#d32f2f")

    breach = retro_df[retro_df['event'] == 'BREACH']
    
    if not breach.empty:
        fig.add_trace(go.Scatter(
            x=breach['date'], y=breach['dii_score'],
            mode='markers', name='ACTUAL BREACH Apr 20 2020',
            marker=dict(size=18, color='#b71c1c', symbol='x',
                        line=dict(width=3, color='#b71c1c')),
        ))

    fig.add_vrect(
        x0="2020-04-07", x1="2020-04-20",
        fillcolor="#ffcdd2", opacity=0.4, line_width=0,
        annotation_text="13-DAY EARLY WARNING WINDOW",
        annotation_position="top left",
        annotation_font_color="#b71c1c",
    )

    fig.update_layout(
        title="Retrospective Simulation — Sasan UMPP 2020 · with Change Point Detection",
        xaxis_title="Date",
        yaxis=dict(range=[0, 1.05]),
        height=440,
        margin=dict(l=10,r=20,t=60,b=80),
        plot_bgcolor='rgba(248,248,248,1)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', yanchor='bottom', y=-0.25),
    )

    fig.add_annotation(
        text="⚠ " + config.RETRO_DISCLAIMER,
        xref="paper", yref="paper",
        x=0, y=-0.22, showarrow=False,
        font=dict(size=10, color="#666"), align="left",
    )
    return fig