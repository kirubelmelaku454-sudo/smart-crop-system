import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Smart Crop Advisor",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Smart Crop Recommendation System")
st.markdown("AI-powered agriculture decision support system for farmers")

# =========================
# LOAD MODEL
# =========================
model = joblib.load("crop_model.pkl")
scaler = joblib.load("crop_scaler.pkl")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("Input Soil & Weather Data")

n = st.sidebar.number_input("Nitrogen (N)", 0, 200, 90)
p = st.sidebar.number_input("Phosphorus (P)", 0, 200, 42)
k = st.sidebar.number_input("Potassium (K)", 0, 300, 43)
temperature = st.sidebar.number_input("Temperature (°C)", 0.0, 60.0, 20.0)
humidity = st.sidebar.number_input("Humidity (%)", 0.0, 100.0, 80.0)
ph = st.sidebar.number_input("pH Level", 0.0, 14.0, 6.5)
rainfall = st.sidebar.number_input("Rainfall (mm)", 0.0, 1000.0, 200.0)

# =========================
# PREDICTION
# =========================
if st.sidebar.button("🌱 Predict Crop"):

    input_data = np.array([[n, p, k, temperature, humidity, ph, rainfall]])

    scaled_data = scaler.transform(input_data)

    prediction = model.predict(scaled_data)[0]

    # Confidence
    confidence = np.max(model.predict_proba(scaled_data)) * 100

    # =========================
    # RESULT DISPLAY
    # =========================
    st.success(f"🌾 Recommended Crop: {prediction.upper()}")

    st.metric("Confidence Level", f"{confidence:.2f}%")

    # =========================
    # IMAGE DISPLAY
    # =========================
    image_path = f"assets/{prediction}.jpg"

    if os.path.exists(image_path):
        st.image(image_path, caption=prediction.upper())
    else:
        st.warning("No image found for this crop")

    # =========================
    # CHART
    # =========================
    chart_data = pd.DataFrame({
        "Feature": ["N", "P", "K"],
        "Value": [n, p, k]
    })

    fig = px.bar(chart_data, x="Feature", y="Value", title="Soil Nutrients")
    st.plotly_chart(fig)

    # =========================
    # FERTILIZER RECOMMENDATION
    # =========================
    fertilizer_map = {
        "rice": "Urea + DAP + Organic Compost",
        "maize": "NPK + Urea",
        "coffee": "Organic Compost + Nitrogen-rich fertilizer",
        "wheat": "DAP + Urea",
        "banana": "Potassium-rich fertilizer"
    }

    st.info("🌱 Fertilizer Recommendation: " + fertilizer_map.get(prediction, "General balanced fertilizer"))

# =========================
# CSV BATCH PREDICTION
# =========================
st.markdown("---")
st.subheader("📂 Batch Prediction (CSV Upload)")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    required_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

    if all(col in data.columns for col in required_cols):

        scaled = scaler.transform(data[required_cols])

        predictions = model.predict(scaled)

        data["Recommended_Crop"] = predictions

        st.success("Prediction completed!")

        st.dataframe(data)

        # Download
        csv = data.to_csv(index=False)

        st.download_button(
            "⬇️ Download Results",
            csv,
            "crop_predictions.csv",
            "text/csv"
        )

    else:
        st.error(f"CSV must contain: {', '.join(required_cols)}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("💡 Smart Agriculture System | Built with Streamlit + Machine Learning")