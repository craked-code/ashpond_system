import folium
import config

COLOR_MAP  = {"CRITICAL":"#d32f2f","ELEVATED":"#f57c00","STABLE":"#388e3c"}
RADIUS_MAP = {"CRITICAL":18,"ELEVATED":14,"STABLE":10}

def build_risk_map(df):
    m = folium.Map(
        location=[config.MAP_CENTER_LAT, config.MAP_CENTER_LON],
        zoom_start=config.MAP_ZOOM,
        tiles="CartoDB positron"
    )
    for _, row in df.iterrows():
        color   = COLOR_MAP.get(row['risk_category'], "#607d8b")
        radius  = RADIUS_MAP.get(row['risk_category'], 10)
        is_anom = bool(row.get('profile_anomaly', False))
        bar     = "█"*int(row['dii_score']*10) + "░"*(10-int(row['dii_score']*10))
        b_prob  = row.get('breach_probability', 0)
        d_lower = row.get('dii_lower', row['dii_score'])
        d_upper = row.get('dii_upper', row['dii_score'])

        popup_html = f"""
            <div style="font-family:Arial;min-width:240px;padding:8px;">
                <h4 style="color:{color};margin:0 0 6px 0;">{row['pond_name']}</h4>
                <b>Plant:</b> {row['plant_name']}<br>
                <hr style="margin:6px 0">
                <b>DII:</b> <span style="font-size:18px;font-weight:bold;color:{color};">
                    {row['dii_score']:.2f} {row.get('dii_trend','→')}
                </span>
                <span style="font-size:11px;color:#666;"> [{d_lower:.2f}–{d_upper:.2f}]</span><br>
                <div style="font-family:monospace;color:{color};">{bar}</div>
                <b>Status:</b> <span style="color:{color};font-weight:bold;">{row['risk_category']}</span><br>
                <b>P(breach):</b> {int(b_prob*100)}%<br>
                {'<b style="color:#e65100">⚠ ANOMALOUS PROFILE</b><br>' if is_anom else ''}
                <hr style="margin:6px 0">
                <b>Top Risk:</b> {row['top_risk_factor']}<br>
                <small>NDWI:{row['ndwi_score']} NDVI:{row['ndvi_score']}
                    Slope:{row['slope_degrees']}° Rain:{row['rainfall_forecast_mm']}mm</small><br>
                <small>SAR backscatter: {row.get('sar_backscatter','N/A')}</small>
            </div>"""
        
        if is_anom and row['risk_category'] == 'STABLE':
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{row['pond_name']} | DII:{row['dii_score']:.2f} | ANOMALOUS",
                icon=folium.Icon(color='orange', icon='exclamation-sign')
            ).add_to(m)
        else:
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                weight=3,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{row['pond_name']} | DII:{row['dii_score']:.2f} | P(breach):{int(b_prob*100)}%"
            ).add_to(m)

    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:8px;
                border:2px solid #ccc;font-family:Arial;font-size:13px;">
        <b>DII Risk Level</b><br>
        <span style="color:#d32f2f;">⬤</span> CRITICAL (DII &gt; 0.70)<br>
        <span style="color:#f57c00;">⬤</span> ELEVATED (0.40–0.70)<br>
        <span style="color:#388e3c;">⬤</span> STABLE (DII &lt; 0.40)<br>
        <span style="color:#e65100;">◆</span> ANOMALOUS PROFILE
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m