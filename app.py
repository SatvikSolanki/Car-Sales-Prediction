import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main { padding-top: 1rem; }
        .app-title {
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0rem;
        }
        .app-subtitle {
            color: #6b7280;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }
        .price-card {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            color: white;
            margin: 1.5rem 0;
            box-shadow: 0 8px 24px rgba(79, 70, 229, 0.25);
        }
        .price-card .label {
            font-size: 1rem;
            opacity: 0.85;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }
        .price-card .value {
            font-size: 3rem;
            font-weight: 800;
            margin-top: 0.25rem;
        }
        div.stButton > button {
            width: 100%;
            background-color: #4f46e5;
            color: white;
            font-weight: 600;
            padding: 0.6rem 0;
            border-radius: 10px;
            border: none;
        }
        div.stButton > button:hover {
            background-color: #4338ca;
            color: white;
        }
        .metric-box {
            background-color: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load model artifacts (cached so this only runs once)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("linear_regression_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    try:
        metrics = joblib.load("metrics.pkl")
    except FileNotFoundError:
        metrics = None
    return model, encoders, feature_columns, metrics


model, encoders, feature_columns, metrics = load_artifacts()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown('<div class="app-title">🚗 Car Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Estimate the market price of a used car with a Linear Regression model.</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar — about / model info
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About this app")
    st.write(
        "This app predicts used car prices using a **Linear Regression** "
        "model trained on real-world car sales data. Missing values were "
        "dropped, engine-size outliers removed, and price was log-transformed "
        "to improve accuracy."
    )

    if metrics:
        st.subheader("📊 Model performance")
        for name, m in metrics.items():
            st.metric(label=name, value=f"R² = {m['r2']:.3f}", delta=f"RMSE = {m['rmse']:.3f}")

    st.divider()
    st.caption("Built with Streamlit • scikit-learn")

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
st.subheader("Enter car details")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", options=list(encoders["Brand"].classes_))
    body = st.selectbox("Body type", options=list(encoders["Body"].classes_))
    engine_type = st.selectbox("Engine type", options=list(encoders["Engine Type"].classes_))
    registration = st.selectbox("Registered?", options=list(encoders["Registration"].classes_))

with col2:
    year = st.slider("Year of manufacture", min_value=1970, max_value=2024, value=2012)
    mileage = st.number_input("Mileage (thousand km)", min_value=0, max_value=1000, value=150, step=5)
    engine_v = st.number_input("Engine volume (liters)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)

st.write("")
predict_clicked = st.button("🔮 Predict Price")

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if predict_clicked:
    input_dict = {
        "Brand": encoders["Brand"].transform([brand])[0],
        "Body": encoders["Body"].transform([body])[0],
        "Mileage": mileage,
        "EngineV": engine_v,
        "Engine Type": encoders["Engine Type"].transform([engine_type])[0],
        "Registration": encoders["Registration"].transform([registration])[0],
        "Year": year,
    }

    # Ensure column order matches exactly what the model was trained on
    input_df = pd.DataFrame([input_dict])[feature_columns]

    log_price_pred = model.predict(input_df)[0]
    price_pred = np.exp(log_price_pred)

    st.markdown(
        f"""
        <div class="price-card">
            <div class="label">Estimated Price</div>
            <div class="value">€{price_pred:,.0f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("See the details behind this prediction"):
        st.write("**Inputs used:**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": ["Brand", "Body", "Mileage", "EngineV", "Engine Type", "Registration", "Year"],
                    "Value": [brand, body, f"{mileage:,} km", f"{engine_v} L", engine_type, registration, year],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"Raw model output (log price): {log_price_pred:.4f} → exp() → €{price_pred:,.2f}"
        )

    st.info(
        "This is a statistical estimate based on historical data, not a guaranteed valuation. "
        "Actual prices vary with condition, location, and market demand.",
        icon="💡",
    )
else:
    st.caption("Fill in the details above and click **Predict Price** to get an estimate.")
