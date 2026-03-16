import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="RainCast · Melbourne",
    page_icon="🌧️",
    layout="wide"
)

import os
if not os.path.exists('model.pkl'):
    with st.spinner('🔄 Training model for first time... this takes a few minutes...'):
        import train_model
        st.rerun()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Epilogue:wght@300;400;500&display=swap');

:root {
    --sky:    #e8f3fb;
    --deep:   #0f3460;
    --mid:    #1a6b9a;
    --accent: #00b4d8;
    --soft:   #cce8f4;
    --text:   #0d2137;
    --muted:  #4a7a99;
    --white:  #ffffff;
    --card:   rgba(255,255,255,0.85);
    --radius: 18px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'Epilogue', sans-serif;
    background: linear-gradient(135deg, #e8f3fb 0%, #d0eaf8 40%, #b8ddf5 100%);
    min-height: 100vh;
    color: var(--text);
}

.main .block-container {
    max-width: 1100px;
    padding: 2rem 2rem 4rem 2rem;
}

/* ── Hero ── */
.hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(120deg, var(--deep) 0%, var(--mid) 100%);
    border-radius: 24px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.hero::before {
    content: '';
    position: absolute;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(0,180,216,0.25) 0%, transparent 70%);
    top: -80px; right: -60px;
    border-radius: 50%;
}

.hero-tag {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.6rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 0.6rem;
}

.hero-title span { color: var(--accent); }

.hero-sub {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.6);
    font-weight: 300;
}

.hero-badge {
    z-index: 2;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 16px;
    padding: 1.2rem 1.8rem;
    text-align: center;
    backdrop-filter: blur(10px);
    min-width: 140px;
}

.hero-badge-num {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--accent);
}

.hero-badge-label {
    font-size: 0.72rem;
    color: rgba(255,255,255,0.55);
    letter-spacing: 1px;
}

/* ── Pills ── */
.pill-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 1.8rem;
}

.pill {
    background: rgba(255,255,255,0.7);
    border: 1px solid var(--soft);
    border-radius: 50px;
    padding: 0.35rem 1rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--mid);
}

/* ── Section titles ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--mid);
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--soft), transparent);
}

/* ── Cards ── */
.input-card {
    background: var(--card);
    border: 1px solid rgba(255,255,255,0.95);
    border-radius: var(--radius);
    padding: 1.6rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 24px rgba(15, 52, 96, 0.08);
}

/* ── Streamlit widget overrides ── */
div[data-testid="stSlider"] label p,
div[data-testid="stSelectbox"] label p {
    color: var(--text) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    font-family: 'Epilogue', sans-serif !important;
}

div[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
}

div[data-testid="stSelectbox"] > div > div {
    background: #f0f8ff !important;
    border: 1.5px solid var(--soft) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'Epilogue', sans-serif !important;
}

div[data-testid="stSelectbox"] > div > div > div {
    color: var(--text) !important;
}

div[data-baseweb="option"] {
    color: var(--text) !important;
    background: white !important;
    font-family: 'Epilogue', sans-serif !important;
}

div[data-baseweb="option"]:hover {
    background: var(--sky) !important;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--deep) 0%, var(--mid) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 1rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    box-shadow: 0 6px 24px rgba(15, 52, 96, 0.3) !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 32px rgba(15, 52, 96, 0.45) !important;
}

/* ── Result cards ── */
.result-wrap {
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: popIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
}

@keyframes popIn {
    from { transform: scale(0.9); opacity: 0; }
    to   { transform: scale(1);   opacity: 1; }
}

.result-rain {
    background: linear-gradient(135deg, #0f3460 0%, #1a6b9a 100%);
    border: 1px solid rgba(0,180,216,0.4);
    box-shadow: 0 16px 48px rgba(15, 52, 96, 0.3);
}

.result-norain {
    background: linear-gradient(135deg, #f5a623 0%, #f7c948 100%);
    border: 1px solid rgba(255,255,255,0.4);
    box-shadow: 0 16px 48px rgba(245, 166, 35, 0.3);
}

.result-icon { font-size: 4rem; margin-bottom: 0.75rem; }

.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.result-rain .result-label  { color: var(--accent); }
.result-norain .result-label { color: rgba(255,255,255,0.8); }

.result-headline {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 1rem;
    color: #ffffff;
}

.result-confidence {
    display: inline-block;
    padding: 0.45rem 1.4rem;
    border-radius: 50px;
    font-size: 0.88rem;
    font-weight: 500;
}

.result-rain .result-confidence {
    background: rgba(0,180,216,0.2);
    color: var(--accent);
    border: 1px solid rgba(0,180,216,0.35);
}

.result-norain .result-confidence {
    background: rgba(255,255,255,0.35);
    color: white;
    border: 1px solid rgba(255,255,255,0.5);
}
  /* Dropdown arrow */
div[data-testid="stSelectbox"] svg {
    fill: var(--text) !important;
    color: var(--text) !important;
    opacity: 1 !important;
}

/* Dropdown border fix */
div[data-testid="stSelectbox"] > div > div {
    background: white !important;
    border: 1.5px solid #a8d8f0 !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
            /* Slider hover tooltip numbers */
div[data-testid="stSlider"] div[data-testid="stTickBarMin"],
div[data-testid="stSlider"] div[data-testid="stTickBarMax"] {
    color: #0d2137 !important;
}

div[data-testid="stSlider"] [class*="StyledThumbValue"] {
    color: #0d2137 !important;
}

            /* Slider thumb value - force black */
div[data-testid="stSlider"] p {
    color: #0d2137 !important;
}

div[data-testid="stSlider"] span {
    color: #0d2137 !important;
}

div[data-testid="stSlider"] div {
    color: #0d2137 !important;
}

/* Navbar / toolbar white text */
header[data-testid="stHeader"] {
    background: linear-gradient(120deg, #0f3460 0%, #1a6b9a 100%) !important;
}

header[data-testid="stHeader"] button,
header[data-testid="stHeader"] a,
header[data-testid="stHeader"] span,
header[data-testid="stHeader"] p {
    color: white !important;
    fill: white !important;
}

header[data-testid="stHeader"] svg {
    fill: white !important;
    color: white !important;
}          
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

try:
    bundle        = load_model()
    model         = bundle['model']
    feature_order = bundle['feature_order']
except FileNotFoundError:
    st.error("⚠️ model.pkl not found. Run `python train_model.py` first.")
    st.stop()

st.markdown("""
<div class="hero">
  <div>
    <div class="hero-tag">🌦️ Machine Learning · Weather Prediction</div>
    <div class="hero-title">Melbourne <span>Rain</span>Cast</div>
    <div class="hero-sub">Powered by RandomForest · Bureau of Meteorology data · 2008–2017</div>
  </div>
  <div class="hero-badge">
    <div class="hero-badge-num">84.5%</div>
    <div class="hero-badge-label">MODEL ACCURACY</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="pill-row">
  <div class="pill">📍 Melbourne</div>
  <div class="pill">📍 MelbourneAirport</div>
  <div class="pill">📍 Watsonia</div>
  <div class="pill">🌿 56k Observations</div>
  <div class="pill">🤖 RandomForest Classifier</div>
</div>
""", unsafe_allow_html=True)

# ── Two-column layout ──
left, right = st.columns([1.05, 1], gap="large")

DIRECTIONS = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']

with left:
   
    st.markdown('<div class="section-title">📍 Location & Season</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        location = st.selectbox("Location", ['Melbourne', 'MelbourneAirport', 'Watsonia'])
    with c2:
        season = st.selectbox("Season", ['Summer', 'Autumn', 'Winter', 'Spring'])

   
    st.markdown('<div class="section-title">🌡️ Temperature</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        min_temp = st.slider("Min Temp (°C)", -5.0, 30.0, 10.0, 0.1)
        temp_9am = st.slider("Temp 9am (°C)", -5.0, 40.0, 15.0, 0.1)
    with c2:
        max_temp = st.slider("Max Temp (°C)", 5.0, 48.0, 22.0, 0.1)
        temp_3pm = st.slider("Temp 3pm (°C)", 5.0, 48.0, 20.0, 0.1)

    
    st.markdown('<div class="section-title">💧 Humidity & Rainfall</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        humidity_9am = st.slider("Humidity 9am (%)", 0, 100, 70)
        rainfall     = st.slider("Rainfall (mm)", 0.0, 100.0, 0.0, 0.1)
    with c2:
        humidity_3pm = st.slider("Humidity 3pm (%)", 0, 100, 50)
        evaporation  = st.slider("Evaporation (mm)", 0.0, 30.0, 5.0, 0.1)

with right:

    st.markdown('<div class="section-title">🌬️ Wind</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        wind_gust_dir   = st.selectbox("Gust Direction", DIRECTIONS)
        wind_gust_speed = st.slider("Gust Speed (km/h)", 0, 130, 35)
        wind_dir_9am    = st.selectbox("Wind Dir 9am", DIRECTIONS)
        wind_speed_9am  = st.slider("Wind Speed 9am (km/h)", 0, 130, 15)
    with c2:
        wind_dir_3pm    = st.selectbox("Wind Dir 3pm", DIRECTIONS)
        wind_speed_3pm  = st.slider("Wind Speed 3pm (km/h)", 0, 130, 20)
        rain_yesterday  = st.selectbox("Rain Yesterday?", ['No', 'Yes'])
    
    
    st.markdown('<div class="section-title">☁️ Pressure, Cloud & Sunshine</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        pressure_9am = st.slider("Pressure 9am (hPa)", 980.0, 1040.0, 1015.0, 0.1)
        cloud_9am    = st.slider("Cloud 9am (oktas)", 0, 8, 4)
        sunshine     = st.slider("Sunshine (hrs)", 0.0, 14.0, 7.0, 0.1)
    with c2:
        pressure_3pm = st.slider("Pressure 3pm (hPa)", 980.0, 1040.0, 1012.0, 0.1)
        cloud_3pm    = st.slider("Cloud 3pm (oktas)", 0, 8, 4)
    

    # ── Predict ──
    predict = st.button("⚡  PREDICT RAINFALL")

    if predict:
        input_dict = {
            'Location': location, 'MinTemp': min_temp, 'MaxTemp': max_temp,
            'Rainfall': rainfall, 'Evaporation': evaporation, 'Sunshine': sunshine,
            'WindGustDir': wind_gust_dir, 'WindGustSpeed': wind_gust_speed,
            'WindDir9am': wind_dir_9am, 'WindDir3pm': wind_dir_3pm,
            'WindSpeed9am': wind_speed_9am, 'WindSpeed3pm': wind_speed_3pm,
            'Humidity9am': humidity_9am, 'Humidity3pm': humidity_3pm,
            'Pressure9am': pressure_9am, 'Pressure3pm': pressure_3pm,
            'Cloud9am': cloud_9am, 'Cloud3pm': cloud_3pm,
            'Temp9am': temp_9am, 'Temp3pm': temp_3pm,
            'RainYesterday': rain_yesterday, 'Season': season
        }

        input_df   = pd.DataFrame([input_dict])[feature_order]
        prediction = model.predict(input_df)[0]
        proba      = model.predict_proba(input_df)[0]
        rain_prob  = proba[list(model.classes_).index('Yes')] * 100

        if prediction == 'Yes':
            st.markdown(f"""
            <div class="result-wrap result-rain">
                <div class="result-icon">🌧️</div>
                <div class="result-label">Prediction Result</div>
                <div class="result-headline">Rain Expected<br>Today</div>
                <div class="result-confidence">{rain_prob:.0f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            no_prob = 100 - rain_prob
            st.markdown(f"""
            <div class="result-wrap result-norain">
                <div class="result-icon">☀️</div>
                <div class="result-label">Prediction Result</div>
                <div class="result-headline">No Rain<br>Expected</div>
                <div class="result-confidence">{no_prob:.0f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)