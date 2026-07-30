import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import numpy as np
import time
import os
import json

# --- 1. Page Config ---
st.set_page_config(page_title="Smart Parking AI", layout="wide", page_icon="⌘")

# --- 2. Advanced CSS Injection (Modern UI/UX) ---
st.markdown("""
    <style>
        /* Hide default Streamlit elements */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Smooth fade-in animation for fluid loading */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .block-container {
            animation: fadeIn 0.6s ease-out;
            padding-top: 2rem !important;
        }

        /* Typography & Headings */
        .main-title {
            font-size: 2.75rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        
        .sub-title {
            color: #a1a1aa;
            font-size: 1.05rem;
            font-weight: 400;
            letter-spacing: 0.01em;
            margin-bottom: 2.5rem;
        }

        /* Modern Metric Cards */
        .metric-card {
            background-color: #18181b; /* Zinc 900 */
            border: 1px solid #27272a; /* Zinc 800 */
            border-radius: 12px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: #3f3f46;
        }

        .metric-label {
            color: #a1a1aa;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
            margin-bottom: 8px;
        }

        .metric-value {
            color: #fafafa;
            font-size: 2.25rem;
            font-weight: 300;
            line-height: 1.2;
            margin: 0;
        }
        
        .metric-sub {
            font-size: 1rem;
            color: #71717a;
            font-weight: 400;
        }
        
        /* Subtle colored dots */
        .dot-green { color: #10b981; margin-right: 6px; }
        .dot-blue { color: #38bdf8; margin-right: 6px; }
        .dot-red { color: #f43f5e; margin-right: 6px; }
        .dot-amber { color: #fbbf24; margin-right: 6px; }
    </style>
""", unsafe_allow_html=True)

# --- Load Models & Data ---
@st.cache_resource
def load_models():
    with open("advanced_ai_models.pkl", "rb") as f:
        return pickle.load(f)

models = load_models()
prophet_model = models["prophet"]
iso_forest = models["isolation_forest"]

try:
    with open('CarParkPos.pkl', 'rb') as f:
        posList = pickle.load(f)
except:
    posList = []

def load_data():
    conn = sqlite3.connect('parking_data.db')
    try:
        df_occ = pd.read_sql_query("SELECT * FROM occupancy", conn)
        df_viol = pd.read_sql_query("SELECT * FROM violations", conn)
    except:
        df_occ = pd.DataFrame(columns=['timestamp', 'available_spots', 'total_spots'])
        df_viol = pd.DataFrame(columns=['timestamp', 'spot_id'])
    conn.close()
    if not df_occ.empty: 
        df_occ['timestamp'] = pd.to_datetime(df_occ['timestamp'])
    return df_occ, df_viol

df, df_violations = load_data()

spot_states = {}
if os.path.exists("spot_states.json"):
    try:
        with open("spot_states.json", "r") as f:
            spot_states = json.load(f)
    except:
        pass

# --- Dashboard Header & Controls ---
header_col, control_col1, control_col2 = st.columns([2.5, 1, 1])

with header_col:
    st.markdown('<div class="main-title">Advanced Parking Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-time spatial analysis and time-series forecasting.</div>', unsafe_allow_html=True)

with control_col1:
    st.write("") 
    st.write("")
    auto_refresh = st.toggle("🔄 Live Sync", value=True)

with control_col2:
    st.write("") 
    if st.button("🗑️ Reset Live Data", use_container_width=True):
        conn = sqlite3.connect('parking_data.db')
        cursor = conn.cursor()
        cursor.execute("DELETE FROM occupancy")
        cursor.execute("DELETE FROM violations")
        conn.commit()
        conn.close()
        if os.path.exists("live_frame.jpg"):
            try: os.remove("live_frame.jpg")
            except: pass
        if os.path.exists("spot_states.json"):
            try: os.remove("spot_states.json")
            except: pass
        st.rerun()

# --- Unsupervised AI Anomaly Check ---
if not df.empty:
    latest_data = df.iloc[-1]
    spots_avail = int(latest_data['available_spots'])
    total_spots = int(latest_data['total_spots'])
    occupancy_rate = ((total_spots - spots_avail) / total_spots) * 100
    total_violations = len(df_violations)
    surge_price = 100 if occupancy_rate > 80 else 50
    
    now_hour = datetime.now().hour
    now_day = datetime.now().weekday()
    anomaly_pred = iso_forest.predict([[now_hour, now_day, spots_avail]])[0]
    
    if anomaly_pred == -1:
        st.error(f"Security Notice: Atypical volume detected for {datetime.now().strftime('%A')} at {now_hour}:00.")
    
    # --- Modern HTML KPI Cards ---
    c1, c2, c3, c4 = st.columns(4)
    
    c1.markdown(f"""
        <div class="metric-card">
            <div class="metric-label"><span class="dot-green">●</span> Available Capacity</div>
            <div class="metric-value">{spots_avail} <span class="metric-sub">/ {total_spots}</span></div>
        </div>
    """, unsafe_allow_html=True)
    
    c2.markdown(f"""
        <div class="metric-card">
            <div class="metric-label"><span class="dot-blue">●</span> Live Occupancy</div>
            <div class="metric-value">{occupancy_rate:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

    c3.markdown(f"""
        <div class="metric-card">
            <div class="metric-label"><span class="dot-red">●</span> Active Violations</div>
            <div class="metric-value" style="color: #f43f5e;">{total_violations}</div>
        </div>
    """, unsafe_allow_html=True)

    c4.markdown(f"""
        <div class="metric-card">
            <div class="metric-label"><span class="dot-amber">●</span> Dynamic Rate</div>
            <div class="metric-value">₹{surge_price}<span class="metric-sub">/hr</span></div>
        </div>
    """, unsafe_allow_html=True)
    
else:
    st.info("Awaiting telemetry stream. Initialize edge compute node (`python step3b_main.py`).")

st.write("") 
st.write("") 

# --- Tabbed Interface ---
tab1, tab2, tab3 = st.tabs(["Spatial Twin", "Predictive Forecast", "Telemetry Analytics"])

ui_colors = {'emerald': '#10b981', 'sky': '#38bdf8', 'rose': '#f43f5e', 'zinc': '#3f3f46', 'bg': 'rgba(0,0,0,0)'}

with tab1:
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.markdown("<h4 style='font-weight: 500; color: #e4e4e7; margin-bottom: 0;'>Interactive Digital Twin</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color: #71717a; font-size: 0.9rem;'>Live spatial blueprint mapping.</p>", unsafe_allow_html=True)
        
        if posList and spot_states:
            x_coords, y_coords, colors, texts = [], [], [], []
            for i, pos in enumerate(posList):
                x_coords.append(pos[0])
                y_coords.append(pos[1]) 
                
                str_i = str(i)
                if str_i in spot_states:
                    state = spot_states[str_i]
                    park_time = int(time.time() - state['since'])
                    if state['state'] == 'empty':
                        colors.append(ui_colors['emerald'])
                        texts.append(f"<b>Spot {i}</b><br>Available")
                    else:
                        if park_time > 15:
                            colors.append(ui_colors['rose'])
                            texts.append(f"<b>Spot {i}</b><br>Violation ({park_time}s)")
                        else:
                            colors.append(ui_colors['zinc'])
                            texts.append(f"<b>Spot {i}</b><br>Occupied ({park_time}s)")
                else:
                    colors.append('#27272a')
                    texts.append(f"Spot {i}<br>Offline")

            fig_twin = go.Figure(data=go.Scatter(
                x=x_coords, y=y_coords, mode='markers',
                marker=dict(size=11, color=colors, symbol='square', line=dict(width=0)), 
                text=texts, hoverinfo='text'
            ))
            fig_twin.update_layout(
                margin=dict(l=0, r=0, t=10, b=0), 
                height=450, 
                yaxis=dict(autorange="reversed", showgrid=False, zeroline=False, visible=False),
                xaxis=dict(showgrid=False, zeroline=False, visible=False),
                plot_bgcolor=ui_colors['bg'], paper_bgcolor=ui_colors['bg'],
                hoverlabel=dict(bgcolor="#18181b", font_size=13, font_family="sans-serif")
            )
            st.plotly_chart(fig_twin, use_container_width=True, config={'displayModeBar': False})
            
    with col_r:
        st.markdown("<h4 style='font-weight: 500; color: #e4e4e7; margin-bottom: 0;'>Edge Computer Vision Feed</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color: #71717a; font-size: 0.9rem;'>Real-time ROI analysis.</p>", unsafe_allow_html=True)
        
        if os.path.exists("live_frame.jpg"):
            try:
                st.markdown("""
                <style>
                    img { border-radius: 12px; border: 1px solid #27272a; object-fit: cover; }
                </style>
                """, unsafe_allow_html=True)
                st.image("live_frame.jpg", use_container_width=True)
            except:
                pass

with tab2:
    st.markdown("<h4 style='font-weight: 500; color: #e4e4e7; margin-bottom: 0;'>Time-Series Forecasting</h4>", unsafe_allow_html=True)
    st.markdown("<p style='color: #71717a; font-size: 0.9rem;'>Predictive availability modeling via Facebook Prophet.</p>", unsafe_allow_html=True)
    
    future = prophet_model.make_future_dataframe(periods=12, freq='h')
    forecast = prophet_model.predict(future)
    
    fig_prophet = go.Figure()
    fig_prophet.add_trace(go.Scatter(
        x=forecast['ds'].tail(48), y=forecast['yhat_upper'].tail(48),
        mode='lines', line=dict(width=0), showlegend=False
    ))
    fig_prophet.add_trace(go.Scatter(
        x=forecast['ds'].tail(48), y=forecast['yhat_lower'].tail(48),
        mode='lines', line=dict(width=0), fillcolor='rgba(56, 189, 248, 0.1)', fill='tonexty', name='Confidence Interval'
    ))
    fig_prophet.add_trace(go.Scatter(
        x=forecast['ds'].tail(48), y=forecast['yhat'].tail(48),
        mode='lines', line=dict(color=ui_colors['sky'], width=3), name='Predicted Trend'
    ))
    
    fig_prophet.update_layout(
        height=400, hovermode="x unified",
        plot_bgcolor=ui_colors['bg'], paper_bgcolor=ui_colors['bg'],
        margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(showgrid=True, gridcolor='#27272a', gridwidth=1, title=""),
        yaxis=dict(showgrid=True, gridcolor='#27272a', gridwidth=1, title="Available Spots"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_prophet, use_container_width=True, config={'displayModeBar': False})

with tab3:
    st.markdown("<h4 style='font-weight: 500; color: #e4e4e7; margin-bottom: 0;'>Historical Traffic Telemetry</h4>", unsafe_allow_html=True)
    
    if not df.empty:
        df['Occupied Spots'] = df['total_spots'] - df['available_spots']
        fig_area = px.area(df, x='timestamp', y='Occupied Spots')
        
        fig_area.update_traces(line_color=ui_colors['emerald'], fillcolor='rgba(16, 185, 129, 0.1)')
        fig_area.update_layout(
            plot_bgcolor=ui_colors['bg'], paper_bgcolor=ui_colors['bg'],
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(showgrid=True, gridcolor='#27272a', title=""),
            yaxis=dict(showgrid=True, gridcolor='#27272a', title="Total Occupied")
        )
        st.plotly_chart(fig_area, use_container_width=True, config={'displayModeBar': False})

if auto_refresh:
    time.sleep(1.5)
    st.rerun()