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

        dii_pct      = int(row['dii_score'] * 100)
        bar_color    = color
        badge_bg     = {"CRITICAL":"#7f1d1d","ELEVATED":"#78350f","STABLE":"#14532d"}.get(row['risk_category'],"#1c2333")
        badge_text   = {"CRITICAL":"#fca5a5","ELEVATED":"#fcd34d","STABLE":"#86efac"}.get(row['risk_category'],"#e2e8f0")
        anom_block   = """
            <div style="background:#431407;border:1px solid #f97316;border-radius:4px;
                        padding:3px 8px;margin:6px 0;font-size:0.72rem;
                        color:#fb923c;font-weight:700;letter-spacing:0.05em;">
                ⚠ ANOMALOUS FACTOR PROFILE
            </div>""" if is_anom else ""

        popup_html = f"""
            <div style="font-family:Inter,Arial,sans-serif;min-width:260px;
                        background:#0f1117;color:#e2e8f0;padding:12px;border-radius:8px;">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                    <div style="font-size:0.95rem;font-weight:700;color:#e2e8f0;">
                        {row['pond_name']}
                    </div>
                    <div style="background:{badge_bg};color:{badge_text};border:1px solid {color};
                                border-radius:4px;padding:2px 8px;font-size:0.7rem;
                                font-weight:700;letter-spacing:0.06em;">
                        {row['risk_category']}
                    </div>
                </div>
                <div style="font-size:0.75rem;color:#64748b;margin-bottom:8px;">
                    {row['plant_name']}
                </div>

                <div style="margin-bottom:6px;">
                    <div style="display:flex;justify-content:space-between;
                                font-size:0.75rem;color:#94a3b8;margin-bottom:3px;">
                        <span>DII Score</span>
                        <span style="color:{color};font-weight:700;">
                            {row['dii_score']:.2f} {row.get('dii_trend','→')}
                            <span style="color:#64748b;font-weight:400;">
                                [{d_lower:.2f}–{d_upper:.2f}]
                            </span>
                        </span>
                    </div>
                    <div style="background:#1c2333;border-radius:4px;height:8px;overflow:hidden;">
                        <div style="width:{dii_pct}%;height:100%;
                                    background:{bar_color};border-radius:4px;">
                        </div>
                    </div>
                </div>

                <div style="display:flex;justify-content:space-between;
                            font-size:0.75rem;margin-bottom:6px;">
                    <span style="color:#94a3b8;">P(Breach 30d)</span>
                    <span style="color:{color};font-weight:700;">{int(b_prob*100)}%</span>
                </div>

                {anom_block}

                <div style="border-top:1px solid #2a2f3e;margin:8px 0;padding-top:8px;
                            font-size:0.73rem;color:#64748b;line-height:1.7;">
                    <div><span style="color:#94a3b8;">Top Risk:</span> {row['top_risk_factor']}</div>
                    <div>NDWI {row['ndwi_score']} · NDVI {row['ndvi_score']} · 
                         Slope {row['slope_degrees']}° · Rain {row['rainfall_forecast_mm']}mm</div>
                    <div>SAR: {row.get('sar_backscatter','N/A')}</div>
                </div>
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
                background:#0f1117;padding:14px 18px;border-radius:8px;
                border:1px solid #2a2f3e;font-family:Inter,Arial,sans-serif;
                font-size:12px;color:#94a3b8;min-width:200px;">
        <div style="font-size:0.75rem;font-weight:700;color:#e2e8f0;
                    letter-spacing:0.06em;margin-bottom:10px;">
            DII RISK LEVEL
        </div>
        <div style="display:flex;flex-direction:column;gap:7px;">
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:12px;height:12px;border-radius:50%;
                            background:#d32f2f;flex-shrink:0;
                            box-shadow:0 0 6px rgba(239,68,68,0.6);"></div>
                <span>CRITICAL <span style="color:#64748b;">(DII &gt; 0.70)</span></span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:12px;height:12px;border-radius:50%;
                            background:#f57c00;flex-shrink:0;"></div>
                <span>ELEVATED <span style="color:#64748b;">(0.40–0.70)</span></span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:12px;height:12px;border-radius:50%;
                            background:#388e3c;flex-shrink:0;"></div>
                <span>STABLE <span style="color:#64748b;">(DII &lt; 0.40)</span></span>
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
                <div style="width:12px;height:12px;border-radius:4px;
                            background:#f97316;flex-shrink:0;"></div>
                <span>ANOMALOUS PROFILE</span>
            </div>
        </div>
    </div>"""
    m.get_root().html.add_child(folium.Element(legend_html))
    return m