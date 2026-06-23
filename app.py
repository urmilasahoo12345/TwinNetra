import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from model import predict_next
import requests
import h5py
import numpy as np
import joblib
from datetime import datetime

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="TwinNetra",
    page_icon="🌦",
    layout="wide"
)

# ==========================================
# ── UI ENHANCEMENT: CSS & Theme ──────────
# ==========================================

# Register custom Plotly theme
TWINNETRA_THEME = {
    "layout": {
        "paper_bgcolor": "#101928",
        "plot_bgcolor": "#101928",
        "font": {"color": "#F0F4FF", "family": "Inter, sans-serif"},
        "colorway": ["#00C9C8", "#FF6B2B", "#FFB830", "#2ED573", "#FF4757", "#7B8CDE"],
        "xaxis": {
            "gridcolor": "rgba(0,201,200,0.08)",
            "linecolor": "rgba(0,201,200,0.2)",
            "tickcolor": "#8A97B0",
            "tickfont": {"color": "#8A97B0", "size": 11},
            "zerolinecolor": "rgba(0,201,200,0.15)"
        },
        "yaxis": {
            "gridcolor": "rgba(0,201,200,0.08)",
            "linecolor": "rgba(0,201,200,0.2)",
            "tickcolor": "#8A97B0",
            "tickfont": {"color": "#8A97B0", "size": 11},
            "zerolinecolor": "rgba(0,201,200,0.15)"
        },
        "title": {"font": {"family": "Rajdhani, sans-serif", "size": 16, "color": "#F0F4FF"}},
        "legend": {"bgcolor": "rgba(16,25,40,0.8)", "bordercolor": "rgba(0,201,200,0.15)", "borderwidth": 1},
        "hoverlabel": {"bgcolor": "#1A2540", "bordercolor": "rgba(0,201,200,0.3)", "font": {"color": "#F0F4FF"}}
    }
}
pio.templates["twinnetra"] = pio.templates["plotly"]
pio.templates["twinnetra"].layout.update(TWINNETRA_THEME["layout"])
pio.templates.default = "twinnetra"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Inter:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root {
  --navy:    #0B1120;
  --navy-2:  #101928;
  --navy-3:  #1A2540;
  --saffron: #FF6B2B;
  --teal:    #00C9C8;
  --amber:   #FFB830;
  --star:    #F0F4FF;
  --muted:   #8A97B0;
  --danger:  #FF4757;
  --success: #2ED573;
  --glow-s:  0 0 16px rgba(0,201,200,0.25);
  --glow-o:  0 0 20px rgba(255,107,43,0.35);
}

html, body, .stApp {
  background-color: var(--navy) !important;
  color: var(--star) !important;
  font-family: 'Inter', sans-serif !important;
}

#MainMenu, footer { visibility: hidden; }

.block-container {
  padding-top: 0.5rem !important;
  max-width: 1300px !important;
}

/* ── Hero Banner ── */
.twinnetra-hero {
  background: linear-gradient(135deg, #0B1120 0%, #1A2540 55%, #0d1f35 100%);
  border: 1px solid rgba(0,201,200,0.18);
  border-radius: 16px;
  padding: 2.2rem 2.8rem 1.8rem;
  margin-bottom: 1.4rem;
  position: relative;
  overflow: hidden;
  box-shadow: 0 0 60px rgba(0,201,200,0.07), inset 0 0 80px rgba(11,17,32,0.5);
}
.twinnetra-hero::before {
  content: '';
  position: absolute; top:0; left:0; right:0; bottom:0;
  background: repeating-linear-gradient(
    0deg,
    rgba(0,201,200,0.022) 0px,
    rgba(0,201,200,0.022) 1px,
    transparent 1px, transparent 4px
  );
  pointer-events: none;
}
.twinnetra-hero::after {
  content: '';
  position: absolute; top:-60px; right:-60px;
  width:320px; height:320px;
  background: radial-gradient(circle, rgba(255,107,43,0.11) 0%, transparent 70%);
  pointer-events: none;
}
.hero-eyebrow {
  font-family: 'Space Mono', monospace;
  font-size: 0.68rem;
  letter-spacing: 0.2em;
  color: var(--teal);
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}
.hero-title {
  font-family: 'Rajdhani', sans-serif;
  font-size: 3.2rem;
  font-weight: 700;
  line-height: 1.05;
  background: linear-gradient(90deg, #F0F4FF 0%, #00C9C8 55%, #FF6B2B 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 0.3rem;
}
.hero-sub {
  font-size: 0.92rem;
  color: var(--muted);
  font-weight: 400;
  letter-spacing: 0.03em;
}
.hero-badges { display: flex; gap: 10px; margin-top: 1rem; flex-wrap: wrap; }
.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: 'Space Mono', monospace;
  font-size: 0.62rem;
  background: rgba(0,201,200,0.08);
  border: 1px solid rgba(0,201,200,0.28);
  color: var(--teal);
  padding: 4px 12px;
  border-radius: 20px;
  letter-spacing: 0.1em;
}
.hero-badge-orange {
  background: rgba(255,107,43,0.08);
  border-color: rgba(255,107,43,0.28);
  color: var(--saffron);
}
.dot-pulse { animation: pulse 1.8s ease infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.25} }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--navy-2) !important;
  border-right: 1px solid rgba(0,201,200,0.1) !important;
}
section[data-testid="stSidebar"] h1 {
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 1.45rem !important;
  font-weight: 700 !important;
  color: var(--star) !important;
}
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] .stMarkdown h3 {
  font-family: 'Rajdhani', sans-serif !important;
  color: var(--teal) !important;
  font-size: 0.85rem !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  border-bottom: 1px solid rgba(0,201,200,0.15) !important;
  padding-bottom: 0.4rem !important;
  margin-bottom: 0.6rem !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
  color: var(--muted) !important;
  font-size: 0.83rem !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {
  color: var(--muted) !important;
  font-size: 0.72rem !important;
  font-family: 'Space Mono', monospace !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: var(--navy-3) !important;
  border: 1px solid rgba(0,201,200,0.18) !important;
  border-radius: 8px !important;
  color: var(--star) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
  background: var(--navy-3) !important;
  border: 1px solid rgba(0,201,200,0.14) !important;
  border-radius: 12px !important;
  padding: 1.1rem 1.3rem !important;
  box-shadow: var(--glow-s) !important;
  transition: border-color 0.22s, box-shadow 0.22s !important;
  position: relative;
  overflow: hidden;
}
[data-testid="metric-container"]:hover {
  border-color: rgba(0,201,200,0.4) !important;
  box-shadow: 0 0 28px rgba(0,201,200,0.22) !important;
}
[data-testid="metric-container"]::before {
  content: '';
  position: absolute; top:0; left:0; right:0; height:2px;
  background: linear-gradient(90deg, var(--teal), var(--saffron));
}
[data-testid="stMetricLabel"] {
  font-family: 'Space Mono', monospace !important;
  font-size: 0.62rem !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 1.9rem !important;
  font-weight: 700 !important;
  color: var(--star) !important;
  line-height: 1.1 !important;
}
[data-testid="stMetricDelta"] {
  font-family: 'Space Mono', monospace !important;
  font-size: 0.72rem !important;
}

/* ── Headings ── */
.stApp h1, .stApp h2 {
  font-family: 'Rajdhani', sans-serif !important;
  font-weight: 700 !important;
  color: var(--star) !important;
  letter-spacing: 0.04em !important;
}
.stApp h3 {
  font-family: 'Rajdhani', sans-serif !important;
  font-weight: 600 !important;
  color: var(--teal) !important;
  letter-spacing: 0.05em !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--navy-2) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  gap: 2px !important;
  border: 1px solid rgba(0,201,200,0.1) !important;
  flex-wrap: wrap !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--muted) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  border-radius: 7px !important;
  padding: 6px 12px !important;
  border: none !important;
  transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(0,201,200,0.14), rgba(255,107,43,0.1)) !important;
  color: var(--star) !important;
  border: 1px solid rgba(0,201,200,0.28) !important;
  box-shadow: 0 0 12px rgba(0,201,200,0.14) !important;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--star) !important; background: rgba(255,255,255,0.04) !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Alerts ── */
.stSuccess { background: rgba(46,213,115,0.07) !important; border-color: var(--success) !important; }
.stWarning { background: rgba(255,184,48,0.07) !important; border-color: var(--amber) !important; }
.stError   { background: rgba(255,71,87,0.07)  !important; border-color: var(--danger) !important; }
.stInfo    { background: rgba(0,201,200,0.06)   !important; border-color: var(--teal) !important; }
.stAlert   { border-radius: 10px !important; }

/* ── Button ── */
.stButton > button {
  font-family: 'Rajdhani', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  letter-spacing: 0.07em !important;
  text-transform: uppercase !important;
  background: linear-gradient(135deg, var(--saffron) 0%, #d44a0b 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.55rem 1.7rem !important;
  box-shadow: 0 0 18px rgba(255,107,43,0.28) !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 0 30px rgba(255,107,43,0.48) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
  background: var(--navy-3) !important;
  border: 1px solid rgba(0,201,200,0.28) !important;
  color: var(--teal) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.72rem !important;
  letter-spacing: 0.06em !important;
  box-shadow: none !important;
  text-transform: none !important;
  font-weight: 400 !important;
}
[data-testid="stDownloadButton"] > button:hover { background: rgba(0,201,200,0.07) !important; }

/* ── Inputs ── */
[data-baseweb="select"] > div,
.stTextInput > div > div,
.stNumberInput > div > div {
  background: var(--navy-3) !important;
  border: 1px solid rgba(0,201,200,0.16) !important;
  border-radius: 8px !important;
  color: var(--star) !important;
}
[data-baseweb="select"] span { color: var(--star) !important; }
.stSelectbox label, .stNumberInput label, .stTextInput label {
  color: var(--muted) !important;
  font-size: 0.72rem !important;
  font-family: 'Space Mono', monospace !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}

/* ── Slider ── */
.stSlider [role="slider"] { background: var(--saffron) !important; border-color: var(--saffron) !important; box-shadow: 0 0 10px rgba(255,107,43,0.45) !important; }
.stSlider label { color: var(--muted) !important; font-family: 'Space Mono', monospace !important; font-size: 0.72rem !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }

/* ── DataFrames ── */
.stDataFrame { border: 1px solid rgba(0,201,200,0.13) !important; border-radius: 10px !important; overflow: hidden !important; }

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid rgba(0,201,200,0.1) !important; margin: 1.4rem 0 !important; }

/* ── Caption ── */
.stCaption { color: var(--muted) !important; font-family: 'Space Mono', monospace !important; font-size: 0.68rem !important; letter-spacing: 0.05em !important; }

/* ── Dropdown menu ── */
[data-baseweb="popover"] ul { background: var(--navy-3) !important; border: 1px solid rgba(0,201,200,0.18) !important; border-radius: 8px !important; }
[data-baseweb="popover"] li { color: var(--star) !important; font-size: 0.86rem !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] [aria-selected="true"] { background: rgba(0,201,200,0.1) !important; color: var(--teal) !important; }

/* ── Code / pre ── */
.stMarkdown pre, .stMarkdown code {
  background: var(--navy-3) !important;
  border: 1px solid rgba(0,201,200,0.14) !important;
  border-radius: 8px !important;
  color: var(--teal) !important;
  font-family: 'Space Mono', monospace !important;
  font-size: 0.8rem !important;
  line-height: 1.9 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--navy-2); }
::-webkit-scrollbar-thumb { background: rgba(0,201,200,0.22); border-radius: 3px; }

/* ── Section divider label ── */
.section-label {
  font-family: 'Space Mono', monospace;
  font-size: 0.62rem;
  color: var(--teal);
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-bottom: 0.6rem;
  opacity: 0.7;
}
</style>

<!-- ── Mission Control Hero Banner ── -->
<div class="twinnetra-hero">
  <div class="hero-eyebrow">ISRO HACKATHON 2026 &nbsp;·&nbsp; CLIMATE INTELLIGENCE SYSTEM &nbsp;·&nbsp; ODISHA, INDIA</div>
  <div class="hero-title">TwinNetra</div>
  <div class="hero-sub">AI-Powered Digital Twin of Odisha's Climate &nbsp;&nbsp;·&nbsp;&nbsp; Historical &nbsp;·&nbsp; Live &nbsp;·&nbsp; Predictive</div>
  <div class="hero-badges">
    <span class="hero-badge"><span class="dot-pulse">●</span> &nbsp;SYSTEM ONLINE</span>
    <span class="hero-badge"><span class="dot-pulse">●</span> &nbsp;INSAT-3DR INTEGRATED</span>
    <span class="hero-badge hero-badge-orange">IMD DATASET 2024</span>
    <span class="hero-badge hero-badge-orange">OPEN-METEO LIVE API</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv("twinnetra_climate_2024.csv")
df["TIME"] = pd.to_datetime(df["TIME"])
df["MONTH"] = df["TIME"].dt.month

# ==========================================
# MOSDAC LST DATA
# ==========================================

try:
    file = "3RIMG_19JUN2026_0015_L2B_LST_V01R00.h5"
    with h5py.File(file, "r") as f:
        lst = f["LST"][0]
        lat = f["Latitude"][:].astype(float)
        lon = f["Longitude"][:].astype(float)
    lat[lat == 32767] = np.nan
    lon[lon == 32767] = np.nan
    lat *= 0.01
    lon *= 0.01
    lst[lst == -999] = np.nan
    mask = (lat >= 17.5) & (lat <= 22.5) & (lon >= 81.5) & (lon <= 87.5)
    odisha_temp = (lst - 273.15)[mask]
    odisha_lat  = lat[mask]
    odisha_lon  = lon[mask]
    avg_lst = np.nanmean(odisha_temp)
    max_lst = np.nanmax(odisha_temp)
    min_lst = np.nanmin(odisha_temp)
    satellite_loaded = True
except Exception as e:
    satellite_loaded = False
    print("MOSDAC ERROR:", e)
    st.error(f"MOSDAC Error: {e}")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🌦 TwinNetra")
st.sidebar.markdown("""
### Climate Control Panel

Features:

- Historical Analysis
- Live Weather Monitoring
- Prediction Engine
- Climate Intelligence
""")

selected_month = st.sidebar.selectbox(
    "Select Month",
    ["All"] + list(range(1, 13))
)
district = st.sidebar.selectbox(
    "Select District",
    ["Bhubaneswar", "Cuttack", "Puri", "Sambalpur", "Rourkela", "Balasore"]
)
future_rise = st.sidebar.slider("Future Temperature Rise (°C)", 0, 5, 2)

# ==========================================
# FILTER DATA
# ==========================================

if selected_month == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[df["MONTH"] == selected_month]

st.info(f"📍 Currently Viewing: **{district}**")

# ==========================================
# KPI SECTION
# ==========================================

st.markdown('<div class="section-label">◈ Climate Overview</div>', unsafe_allow_html=True)
st.header("Climate Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Rainfall",  f"{filtered_df['RAINFALL'].mean():.2f} mm")
col2.metric("Avg Temp",      f"{filtered_df['MAX_TEMP'].mean():.2f} °C")
col3.metric("Max Rainfall",  f"{filtered_df['RAINFALL'].max():.2f} mm")
col4.metric("Max Temp",      f"{filtered_df['MAX_TEMP'].max():.2f} °C")

st.markdown("---")

# ==========================================
# RISK INDICATOR
# ==========================================

heatwaves = len(filtered_df[filtered_df["MAX_TEMP"] >= 40])

if heatwaves > 100:
    st.error(f"⚠ High Heatwave Risk ({heatwaves} records)")
elif heatwaves > 20:
    st.warning(f"⚠ Moderate Heatwave Risk ({heatwaves} records)")
else:
    st.success(f"✓ Low Heatwave Risk ({heatwaves} records)")

st.markdown("---")

# Load trained models
temp_model = joblib.load("temperature_model.pkl")
rain_model = joblib.load("rainfall_model.pkl")

# ==========================================
# TABS
# ==========================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "Overview", "Rainfall", "Temperature", "Climate Relation",
    "Raw Data", "Climate Map", "Prediction", "Live Climate Monitor",
    "System Architecture", "Satellite LST"
])

# ── Overview ──
with tab1:
    st.subheader("Climate Trend")
    daily = filtered_df.groupby("TIME").agg({"RAINFALL": "mean", "MAX_TEMP": "mean"}).reset_index()
    fig = px.line(daily, x="TIME", y=["RAINFALL", "MAX_TEMP"], title="Odisha Climate Trend (2024)")
    fig.update_layout(xaxis_title="Date", yaxis_title="Climate Value", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    monthly_rain = filtered_df.groupby("MONTH")["RAINFALL"].mean().reset_index()
    monthly_temp = filtered_df.groupby("MONTH")["MAX_TEMP"].mean().reset_index()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(monthly_rain, x="MONTH", y="RAINFALL", title="Monthly Average Rainfall")
        fig.update_layout(xaxis_title="Month", yaxis_title="Rainfall (mm)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(monthly_temp, x="MONTH", y="MAX_TEMP", title="Monthly Average Temperature")
        fig.update_layout(xaxis_title="Month", yaxis_title="Temperature (°C)")
        st.plotly_chart(fig, use_container_width=True)

# ── Rainfall ──
with tab2:
    st.subheader("Rainfall Analytics")
    rain_daily = filtered_df.groupby("TIME")["RAINFALL"].mean().reset_index()
    fig = px.line(rain_daily, x="TIME", y="RAINFALL", title="Daily Rainfall Trend")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Top 10 Rainfall Events")
    st.dataframe(filtered_df.nlargest(10, "RAINFALL"))

# ── Temperature ──
with tab3:
    st.subheader("Temperature Analytics")
    temp_daily = filtered_df.groupby("TIME")["MAX_TEMP"].mean().reset_index()
    fig = px.line(temp_daily, x="TIME", y="MAX_TEMP", title="Daily Temperature Trend")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Top 10 Hottest Events")
    st.dataframe(filtered_df.nlargest(10, "MAX_TEMP"))

# ── Climate Relation ──
with tab4:
    st.subheader("Rainfall vs Temperature")
    fig = px.scatter(filtered_df, x="MAX_TEMP", y="RAINFALL", opacity=0.5, title="Rainfall vs Temperature")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Correlation Matrix")
    corr = filtered_df[["RAINFALL", "MAX_TEMP"]].corr()
    st.dataframe(corr)

# ── Raw Data ──
with tab5:
    st.subheader("Dataset Preview")
    st.dataframe(filtered_df.head(100))
    csv = filtered_df.to_csv(index=False)
    st.download_button("Download Filtered Dataset", csv, "twinnetra_filtered.csv", "text/csv")

# ── Climate Map ──
with tab6:
    st.subheader("🌍 Odisha Climate Heatmap (Digital Twin View)")
    map_option = st.selectbox("Select Variable", ["RAINFALL", "MAX_TEMP"])
    spatial_data = filtered_df.groupby(["LATITUDE", "LONGITUDE"])[map_option].mean().reset_index()
    spatial_data = spatial_data.dropna()
    st.markdown(f"### Heatmap for {map_option}")
    fig = px.density_heatmap(
        spatial_data, x="LONGITUDE", y="LATITUDE", z=map_option,
        color_continuous_scale="Turbo", title=f"Odisha {map_option} Heatmap (2024)"
    )
    fig.update_layout(xaxis_title="Longitude", yaxis_title="Latitude")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("🔥 Top High Intensity Grid Points")
    st.dataframe(spatial_data.sort_values(map_option, ascending=False).head(10))

# ── Prediction ──
with tab7:
    st.subheader("AI Climate Prediction Engine")
    district_coords = {
        "Bhubaneswar": (20.2961, 85.8245),
        "Cuttack":     (20.4625, 85.8828),
        "Puri":        (19.8135, 85.8312),
        "Sambalpur":   (21.4669, 83.9812),
        "Rourkela":    (22.2604, 84.8536),
        "Balasore":    (21.4942, 86.9317)
    }
    district = st.selectbox("Select District", list(district_coords.keys()))
    rainfall    = st.number_input("Current Rainfall (mm)", value=10.0)
    temperature = st.number_input("Current Temperature (°C)", value=30.0)
    month = st.slider("Month", 1, 12, datetime.now().month)
    day   = st.slider("Day",   1, 31, datetime.now().day)

    if st.button("Predict Next Day Climate"):
        lat, lon = district_coords[district]
        predicted_temp = temp_model.predict([[lat, lon, month, day, rainfall]])[0]
        predicted_rain = rain_model.predict([[lat, lon, month, day, temperature]])[0]
        st.success("Prediction Completed")
        c1, c2 = st.columns(2)
        c1.metric("Predicted Temperature", f"{predicted_temp:.2f} °C")
        c2.metric("Predicted Rainfall",    f"{predicted_rain:.2f} mm")
        st.markdown("---")
        st.subheader("Prediction Risk Assessment")
        if predicted_temp > 40:    st.error("Heatwave Risk Detected")
        elif predicted_temp > 37:  st.warning("High Temperature Warning")
        else:                      st.success("Normal Temperature Conditions")
        if predicted_rain > 100:   st.error("Extreme Rainfall Risk")
        elif predicted_rain > 50:  st.warning("Heavy Rainfall Possible")
        else:                      st.success("Normal Rainfall Conditions")
        st.markdown("---")
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("Temperature Model R²", "0.97")
        col2.metric("Rainfall Model R²",    "0.31")
        st.info("Models trained using IMD 2024 Climate Dataset and Random Forest Regressor.")

# ── Live Climate Monitor ──
with tab8:
    st.subheader("Live vs Historical Climate Comparison")
    districts = {
        "Bhubaneswar": (20.2961, 85.8245),
        "Cuttack":     (20.4625, 85.8828),
        "Puri":        (19.8135, 85.8312),
        "Sambalpur":   (21.4669, 83.9812),
        "Rourkela":    (22.2604, 84.8536),
        "Balasore":    (21.4942, 86.9317)
    }
    current_month  = pd.Timestamp.now().month
    historical_df  = df[df["TIME"].dt.month == current_month]
    results = []
    for district, (lat, lon) in districts.items():
        try:
            url = (f"https://api.open-meteo.com/v1/forecast"
                   f"?latitude={lat}&longitude={lon}"
                   f"&current=temperature_2m,relative_humidity_2m,rain")
            data       = requests.get(url).json()
            live_temp  = data["current"]["temperature_2m"]
            humidity   = data["current"]["relative_humidity_2m"]
            rain       = data["current"]["rain"]
            nearest    = historical_df[
                (abs(historical_df["LATITUDE"] - lat) <= 1.0) &
                (abs(historical_df["LONGITUDE"] - lon) <= 1.0)
            ]
            historical_temp = nearest["MAX_TEMP"].mean()
            difference      = live_temp - historical_temp
            status = "Above Normal" if difference > 2 else ("Below Normal" if difference < -2 else "Normal")
            results.append({
                "District": district, "Live Temp": round(live_temp, 2),
                "Historical Temp": round(historical_temp, 2), "Difference": round(difference, 2),
                "Humidity": humidity, "Rainfall": rain, "Status": status
            })
        except:
            pass
    live_df = pd.DataFrame(results)
    st.dataframe(live_df, use_container_width=True)
    fig = px.bar(live_df, x="District", y="Difference", color="Status",
                 title="Temperature Anomaly by District",
                 color_discrete_map={"Above Normal": "#FF6B2B", "Below Normal": "#00C9C8", "Normal": "#2ED573"})
    st.plotly_chart(fig, use_container_width=True)
    st.success("Digital Twin Comparison: Live Weather vs Historical Climate")

# ── System Architecture ──
with tab9:
    st.header("TwinNetra Architecture")
    st.markdown("""
    IMD Rainfall Dataset
            ↓

    IMD Temperature Dataset
            ↓

    Data Processing Layer
            ↓

    Climate Analytics Engine
            ↓

    Live Weather API
            ↓

    Prediction Engine
            ↓

    TwinNetra Dashboard
    """)
    st.success("TwinNetra integrates Historical + Live + Predictive Climate Intelligence")

# ── Satellite LST ──
with tab10:
    st.subheader("INSAT-3DR Land Surface Temperature")
    if satellite_loaded:
        col1, col2, col3 = st.columns(3)
        col1.metric("Average LST", f"{avg_lst:.2f} °C")
        col2.metric("Maximum LST", f"{max_lst:.2f} °C")
        col3.metric("Minimum LST", f"{min_lst:.2f} °C")
        sat_df = pd.DataFrame({"Latitude": odisha_lat, "Longitude": odisha_lon, "LST": odisha_temp})
        fig = px.scatter(sat_df, x="Longitude", y="Latitude", color="LST",
                         title="Odisha Satellite Land Surface Temperature",
                         color_continuous_scale="Turbo")
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.subheader("Ground vs Satellite Comparison")
        try:
            ground_temp = live_df[live_df["District"] == "Bhubaneswar"]["Live Temp"].iloc[0]
            temp_diff   = avg_lst - ground_temp
            c1, c2, c3 = st.columns(3)
            c1.metric("Ground Temperature", f"{ground_temp:.2f} °C")
            c2.metric("Satellite LST",      f"{avg_lst:.2f} °C")
            c3.metric("Difference",         f"{temp_diff:.2f} °C")
        except Exception as e:
            st.warning(f"Live weather data unavailable: {e}")
        st.markdown("---")
        st.subheader("Climate Scenario Simulator")
        future_lst = avg_lst + future_rise
        st.metric("Projected Odisha LST", f"{future_lst:.2f} °C", f"+{future_rise} °C")
        scenario_df = pd.DataFrame({
            "Scenario":    ["Current", f"+{future_rise}°C Future"],
            "Temperature": [avg_lst,   future_lst]
        })
        fig2 = px.bar(scenario_df, x="Scenario", y="Temperature",
                      title="Future Climate Scenario",
                      color="Scenario",
                      color_discrete_map={"Current": "#00C9C8", f"+{future_rise}°C Future": "#FF6B2B"})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("MOSDAC LST file not found.")

# ==========================================
# INSIGHTS SECTION
# ==========================================

st.markdown("---")
st.header("Climate Insights")

wettest_month = df.groupby("MONTH")["RAINFALL"].mean().idxmax()
hottest_month = df.groupby("MONTH")["MAX_TEMP"].mean().idxmax()

st.info(f"""
🌧 Wettest Month : {wettest_month}

☀ Hottest Month : {hottest_month}

🌦 Rainfall-Temperature Correlation : {df[['RAINFALL','MAX_TEMP']].corr().iloc[0,1]:.3f}
""")

st.markdown("---")

if satellite_loaded:
    st.success(f"""
🛰 Satellite Average LST : {avg_lst:.2f} °C

🛰 Satellite Maximum LST : {max_lst:.2f} °C

🛰 INSAT-3DR MOSDAC Data Successfully Integrated
""")

st.markdown("""
## TwinNetra

AI-Powered Digital Twin of Odisha Climate

Data Sources:

- IMD Rainfall Dataset
- IMD Maximum Temperature Dataset
- Open-Meteo Live API

""")