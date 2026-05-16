import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(
    page_title="India Electricity Demand Forecast",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #F0F2F5; }
h1 { font-family: 'IBM Plex Mono', monospace; color: #1C2B33; font-size: 1.8rem; }
h2, h3 { color: #1C2B33; }
p, li, label, div { color: #1C2B33; }
.metric-card {
    background: white;
    border-left: 4px solid #E07B00;
    padding: 1rem 1.2rem;
    border-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #E07B00;
}
.metric-label { font-size: 0.78rem; color: #6B7C85; text-transform: uppercase; letter-spacing: 0.05em; }
.stButton > button {
    background-color: #36454F; color: white; border: none;
    border-radius: 3px; padding: 0.5rem 2rem;
    font-weight: 600; width: 100%;
}
.stButton > button:hover { background-color: #E07B00; color: white; }
div[data-testid="stSidebar"] { background-color: #36454F; }
div[data-testid="stSidebar"] * { color: white !important; }
div[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] { background-color: #4a5f6a !important; }
div[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * { color: white !important; }
div[data-testid="stSidebar"] .stSelectbox svg { fill: white !important; }
div[data-testid="stSidebar"] .stSlider * { color: white !important; }
div[data-testid="stSidebar"] button { background-color: #E07B00 !important; color: white !important; border: none !important; }
div[data-testid="stSidebar"] button:hover { background-color: #c96a00 !important; }
div[data-testid="stSidebar"] .stSelectbox > div > div { background-color: #4a5f6a !important; border: 1px solid #6b8a9a !important; }
div[data-testid="stSidebar"] .stSelectbox > div > div > div { color: white !important; }
div[data-testid="stSidebar"] [data-testid="stButton"] > button { background-color: #E07B00 !important; color: white !important; border: none !important; font-weight: 600 !important; }
div[data-testid="stSidebar"] [data-testid="stSlider"] p { color: white !important; }
div[data-testid="stSidebar"] [data-testid="stSlider"] span { color: white !important; }
div[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMin"] { color: white !important; }
div[data-testid="stSidebar"] [data-testid="stSlider"] div[data-testid="stTickBarMax"] { color: white !important; }
div[data-testid="stSidebar"] p { color: white !important; }
div[data-testid="stSidebar"] label { color: #A8BECA !important; }
div[data-testid="stSidebar"] span { color: white !important; }
div[data-testid="stSidebar"] strong { color: white !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    raw = pd.read_csv("Daily_Power_Gen_Source_march_23.csv", encoding='utf-8-sig')
    raw.columns = raw.columns.str.strip()
    df = raw[raw['source'] == 'Total'].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    return df


@st.cache_data
def get_monthly(col):
    df = load_data()
    monthly = df[col].resample('MS').mean().dropna()
    monthly.loc['2020-04-01'] = (monthly.loc['2020-03-01'] + monthly.loc['2020-05-01']) / 2
    return monthly


@st.cache_data
def run_sarima(col, horizon):
    monthly = get_monthly(col)
    fit  = SARIMAX(monthly, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    fc   = fit.get_forecast(steps=horizon)
    pred = fc.predicted_mean
    ci   = fc.conf_int()
    last_date  = monthly.index[-1]
    pred.index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq='MS')
    ci.index   = pred.index
    return monthly, pred, ci


REGIONS = {
    'All India'       : 'All India',
    'North (NR)'      : 'NR',
    'West (WR)'       : 'WR',
    'South (SR)'      : 'SR',
    'East (ER)'       : 'ER',
    'North-East (NER)': 'NER'
}

with st.sidebar:
    st.markdown("<h3 style='color:white'>⚡ Configuration</h3>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#4a5f6a'>", unsafe_allow_html=True)
    region_label = st.selectbox("Select Region", list(REGIONS.keys()))
    horizon      = st.slider("Forecast Horizon (months)", 3, 24, 12, 3)
    st.markdown(f"<p style='color:white;font-size:0.85rem;margin-top:-10px'>Selected: <b style='color:#E07B00'>{horizon} months</b></p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#4a5f6a'>", unsafe_allow_html=True)
    st.markdown("<p style='color:#A8BECA;font-size:0.85rem'><b style='color:white'>Model:</b> SARIMA(1,1,1)(1,1,1,12)</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#A8BECA;font-size:0.85rem'><b style='color:white'>Data:</b> POSOCO 2013-2023</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:#A8BECA;font-size:0.85rem'><b style='color:white'>Frequency:</b> Monthly</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#4a5f6a'>", unsafe_allow_html=True)
    run = st.button("Run Forecast")

st.title("India Electricity Demand Forecast")
st.markdown(f"Region: **{region_label}** &nbsp;|&nbsp; Horizon: **{horizon} months** &nbsp;|&nbsp; Model: **SARIMA**")

st.info(
    "**How to use this app:**\n\n"
    "1. Select a region from the sidebar\n"
    "2. Choose how many months ahead to forecast\n"
    "3. Click Run Forecast\n"
    "4. Read the forecast table for exact values and confidence bounds"
)

st.markdown("---")

if run or 'done' not in st.session_state:
    with st.spinner("Running SARIMA model..."):
        monthly, pred, ci = run_sarima(REGIONS[region_label], horizon)
    st.session_state.update({'monthly': monthly, 'pred': pred, 'ci': ci, 'done': True})

monthly = st.session_state['monthly']
pred    = st.session_state['pred']
ci      = st.session_state['ci']

peak_month = pred.idxmax().strftime('%b %Y')
change     = ((pred.iloc[0] - monthly.iloc[-1]) / monthly.iloc[-1]) * 100

c1, c2, c3, c4 = st.columns(4)
cards = [
    (c1, "Latest Actual",       f"{monthly.iloc[-1]:,.0f}", "MU / month"),
    (c2, "Forecast Next Month", f"{pred.iloc[0]:,.0f}",     "MU / month"),
    (c3, "Month-on-Month",      f"{change:+.1f}%",           "vs last actual"),
    (c4, "Forecast Peak",       f"{pred.max():,.0f}",        f"MU in {peak_month}"),
]
for col, label, value, sub in cards:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor('#F0F2F5')
ax.set_facecolor('white')
history_plot = monthly.iloc[-36:]
ax.plot(history_plot.index, history_plot.values, color='#36454F', linewidth=2, label='Historical')
ax.plot(pred.index, pred.values, color='#E07B00', linewidth=2, linestyle='--', label='Forecast')
ax.fill_between(pred.index, ci.iloc[:,0], ci.iloc[:,1], color='#E07B00', alpha=0.15, label='95% Confidence Interval')
ax.axvline(monthly.index[-1], color='gray', linestyle=':', linewidth=1)
ax.text(monthly.index[-1], ax.get_ylim()[1]*0.97, ' Forecast start', fontsize=8, color='gray')
ax.set_title(f'Electricity Demand Forecast — {region_label} ({horizon} months ahead)',
             fontsize=13, fontweight='bold', color='#1C2B33')
ax.set_xlabel('Month', color='#36454F')
ax.set_ylabel('Demand (Million Units)', color='#36454F')
ax.tick_params(colors='#36454F')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

st.markdown("### Forecast Values")
forecast_df = pd.DataFrame({
    'Month'           : pred.index.strftime('%b %Y'),
    'Forecast (MU)'   : pred.values.round(1),
    'Lower Bound (MU)': ci.iloc[:,0].values.round(1),
    'Upper Bound (MU)': ci.iloc[:,1].values.round(1),
})
st.dataframe(forecast_df.reset_index(drop=True), use_container_width=True)

st.markdown("### Full Historical Series")
fig2, ax2 = plt.subplots(figsize=(13, 3.5))
fig2.patch.set_facecolor('#F0F2F5')
ax2.set_facecolor('white')
ax2.plot(monthly.index, monthly.values, color='#36454F', linewidth=1.2)
ax2.axvline(pd.Timestamp('2020-03-24'), color='#E07B00', linestyle='--', linewidth=1)
ax2.text(pd.Timestamp('2020-05-01'), monthly.max()*0.96, 'COVID-19', fontsize=8, color='#E07B00')
ax2.set_title(f'Full Historical Demand — {region_label} (2013-2023)', fontsize=11, color='#1C2B33', fontweight='bold')
ax2.set_xlabel('Year', color='#36454F')
ax2.set_ylabel('Demand (MU)', color='#36454F')
ax2.tick_params(colors='#36454F')
ax2.grid(True, alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

st.markdown("---")
st.markdown(
    "<p style='color:#6B7C85;font-size:0.8rem;'>Data: POSOCO via Kaggle · Model: SARIMA(1,1,1)(1,1,1,12) · Dataset: Mar 2013 - Mar 2023</p>",
    unsafe_allow_html=True
)
