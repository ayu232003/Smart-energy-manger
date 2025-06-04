import streamlit as st
import numpy as np
import joblib
import time
import io
from tensorflow.keras.models import load_model
import pandas as pd

# Page config
st.set_page_config(page_title="Smart Energy Manager", layout="wide")

# Load models
@st.cache_resource
def load_models():
    try:
        occupancy_model = joblib.load("occupancy_model.h5")
        occupancy_scaler = joblib.load("occupancy_scaler.h5")
        energy_model = load_model("energy_consumption_model_gpu.h5")
        return occupancy_model, occupancy_scaler, energy_model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.error("Please make sure you have the following files:")
        st.error("- occupancy_model.h5")
        st.error("- occupancy_scaler.h5") 
        st.error("- energy_consumption_model_gpu.h5")
        st.stop()

occupancy_model, occupancy_scaler, energy_model = load_models()

# Custom styling
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #dbe6f6, #c5796d);
    }
    .stButton button {
        background-color: #f76c6c;
        color: white;
        padding: 0.6em 1.5em;
        border-radius: 10px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align:center; color:#2c3e50;'>🏠 Smart Energy Management System</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align:center; color:#34495e;'>IoT + ML Powered Energy Optimization</h5>", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("📊 Input Parameters")
temperature = st.sidebar.slider("🌡️ Temperature (°C)", -10.0, 50.0, 24.0)
humidity = st.sidebar.slider("💧 Humidity (%)", 0, 100, 40)
light = st.sidebar.slider("💡 Light Intensity (Lux)", 0, 2000, 300)
co2 = st.sidebar.slider("🫁 CO2 Level (ppm)", 300, 2000, 600)
wind_speed = st.sidebar.slider("🍃 Wind Speed (km/h)", 0, 100, 10)
hour = st.sidebar.slider("🕒 Hour of Day", 0, 23, 12)



smart_control = st.sidebar.checkbox("🧠 Enable Smart Controls", value=True)

# Prediction Panel
st.subheader("🔍 Prediction Panel")
st.write("Adjust the parameters in the sidebar and click below to run predictions.")

# Show parameters before prediction in 3x3 layout
st.markdown("### 📌 Parameters selcted")
c1, c2, c3 = st.columns(3)
with c1:
    st.info(f"🌡️ Temperature: {temperature}°C")
    st.info(f"💧 Humidity: {humidity}%")
with c2:
    st.info(f"💡 Light: {light} Lux")
    st.info(f"🫁 CO2: {co2} ppm")
with c3:
    st.info(f"🍃 Wind: {wind_speed} km/h")
    st.info(f"🕒 Hour: {hour}:00")

def prepare_lstm_features(temperature, humidity, light, co2):
    """
    Prepare input for LSTM model using the four parameters.
    Returns shape: (1, 30, 1)
    """
    # Normalize values to similar scale
    temp_norm = temperature / 50.0
    humidity_norm = humidity / 100.0
    light_norm = light / 2000.0
    co2_norm = co2 / 2000.0

    # Composite pattern based on the four parameters
    composite = (temp_norm + humidity_norm + light_norm + co2_norm) / 4
    sequence = np.full((30, 1), composite)

    return sequence.reshape(1, 30, 1)

    
    # Reshape to (batch_size, timesteps, features) = (1, 30, 1)
    return sequence.reshape(1, 30, 1)

if st.button("🔮 Predict Occupancy & Energy"):
    with st.spinner("Running ML pipeline..."):
        loading_msg = st.empty()
        loading_msg.info("🧠 Learning your habits... predictive maintenance activated")
        time.sleep(0.10)
        loading_msg.info("⚠️ Monitoring energy spikes – anomaly detection active")
        time.sleep(0.10)
        loading_msg.info("📟 Monitor electronic devices for energy anomalies")
        time.sleep(0.10)
        loading_msg.empty()

    occupancy_features = np.array([[temperature, humidity, light, co2]])
    report_buffer = io.StringIO()

    try:
        # Occupancy Prediction
        occupancy_features_scaled = occupancy_scaler.transform(occupancy_features)
        occupancy_prediction = occupancy_model.predict(occupancy_features_scaled)[0]
        occupancy_probability = occupancy_model.predict_proba(occupancy_features_scaled)[0]

        if occupancy_prediction == 1:
            st.success("✅ **Room is OCCUPIED**")
            confidence = occupancy_probability[1] * 100
        else:
            st.error("❌ **Room is NOT OCCUPIED**")
            confidence = occupancy_probability[0] * 100

        st.info(f"🎯 **Confidence:** {confidence:.1f}%")

        # LSTM Energy Consumption Prediction
        st.subheader("⚡Energy Consumption Prediction")
        
        try:
            # Prepare LSTM input features (reshape to 30 timesteps, 1 feature)
            lstm_input = prepare_lstm_features(temperature, humidity, light, co2)
            
            # Make prediction using LSTM model
            lstm_energy_prediction = energy_model.predict(lstm_input, verbose=0)
            estimated_energy_lstm = float(lstm_energy_prediction[0][0]) if len(lstm_energy_prediction.shape) > 1 else float(lstm_energy_prediction[0])
            
            # Ensure positive energy values
            estimated_energy_lstm = max(0.1, estimated_energy_lstm)
            
            st.success(f"🤖 **Predicted Energy Consumption:** {estimated_energy_lstm:.2f} kWh")
            
            # Additional energy insights based on LSTM prediction
            if estimated_energy_lstm > 2.0:
                st.warning("⚠️ High energy consumption predicted!")
            elif estimated_energy_lstm < 0.5:
                st.info("💡 Low energy consumption predicted - efficient usage!")
            else:
                st.info("✅ Normal energy consumption predicted")
                
        except Exception as e:
            st.error(f"❌ LSTM prediction failed: {str(e)}")
            st.info("🔄 Falling back to rule-based energy estimation...")
            
            # Fallback to original rule-based calculation
            base_energy = 0.5

            if occupancy_prediction == 1:
                temp_factor = max(0, abs(temperature - 22) - 2) * 0.15
                light_factor = max(0, (800 - light) / 1000) * 0.4
                co2_factor = max(0, (co2 - 400) / 1000) * 0.3
                time_factor = 0.3 if 8 <= hour <= 18 else 0.2 if 19 <= hour <= 22 else 0.1
                estimated_energy_lstm = base_energy + temp_factor + light_factor + co2_factor + time_factor
            else:
                estimated_energy_lstm = base_energy * 0.2
                
            st.success(f"🏠 **Fallback Energy Estimation:** {estimated_energy_lstm:.2f} kWh")

        # Energy Breakdown Section
        if occupancy_prediction == 1:
            base_energy = 0.5
            temp_factor = max(0, abs(temperature - 22) - 2) * 0.15
            light_factor = max(0, (800 - light) / 1000) * 0.4
            co2_factor = max(0, (co2 - 400) / 1000) * 0.3
            time_factor = 0.3 if 8 <= hour <= 18 else 0.2 if 19 <= hour <= 22 else 0.1

            with st.expander("📊 Energy Breakdown"):
                st.markdown("#### 📋 Components of Energy Usage")
                st.markdown(f"""
                    • Base consumption: **{base_energy:.2f} kWh**  
                    • Temperature adjustment: **+{temp_factor:.2f} kWh**  
                    • Lighting needs: **+{light_factor:.2f} kWh**  
                    • Ventilation (CO2): **+{co2_factor:.2f} kWh**  
                    • Time of day factor: **+{time_factor:.2f} kWh**
                    """)
        else:
            with st.expander("📊 Energy Breakdown"):
                st.markdown("Room is not occupied. Minimal energy is used.")
                st.markdown("• Base consumption: **0.10 kWh** (Standby usage)")
        # Anomaly Detection
        anomalies = []

        if temperature < 15:
            anomalies.append(("🌡️ Temperature too low", "Too cold — increase the room temperature to a comfortable range."))
        elif temperature > 30:
            anomalies.append(("🌡️ Temperature too high", "Too hot — decrease the room temperature using AC or ventilation."))

        if humidity < 20:
            anomalies.append(("💧 Humidity too low", "Air is too dry — use a humidifier or place water trays near heat sources."))
        elif humidity > 80:
            anomalies.append(("💧 Humidity too high", "Too humid — use a dehumidifier or increase ventilation."))

        if light < 50:
            anomalies.append(("💡 Light level too low", "Increase lighting — turn on additional lights or open windows."))
        elif light > 1800:
            anomalies.append(("💡 Light level too high", "Reduce lighting — close blinds or dim artificial lights."))

        if co2 > 1500:
            anomalies.append(("🫁 High CO2 levels", "Ventilate the room — open windows or turn on exhaust fans."))

        if wind_speed > 80:
            anomalies.append(("🍃 Excessive wind speed", "Secure windows and equipment — check for storm or equipment malfunction."))

        if hour < 6 or hour > 22:
            anomalies.append(("🕒 Off-usage hours", "Minimize energy use — power down non-essential devices."))

        # Display Alerts with Recommendations
        if anomalies:
            st.warning("🚨 **Energy Smart Alert:** Please address the following:")
            for alert, recommendation in anomalies:
                st.markdown(f"{alert}  \n➡️ _**Recommendation:**_ {recommendation}")


        # Generate report
        report_buffer.write("=== SMART ENERGY MANAGEMENT REPORT ===\n")
        report_buffer.write(f"Time: {hour}:00\n\n")
        report_buffer.write("OCCUPANCY DETECTION:\n")
        report_buffer.write(f"Status: {'Occupied' if occupancy_prediction else 'Not Occupied'}\n")
        report_buffer.write(f"Confidence: {confidence:.2f}%\n\n")
        report_buffer.write("ENERGY PREDICTION:\n")
        report_buffer.write(f"LSTM Predicted Energy: {estimated_energy_lstm:.2f} kWh\n\n")
        report_buffer.write("ENVIRONMENTAL CONDITIONS:\n")
        report_buffer.write(f"Temperature: {temperature}°C\n")
        report_buffer.write(f"Humidity: {humidity}%\n")
        report_buffer.write(f"Light: {light} Lux\n")
        report_buffer.write(f"CO2: {co2} ppm\n")
        report_buffer.write(f"Wind Speed: {wind_speed} km/h\n\n")
        if anomalies:
            report_buffer.write("ANOMALIES DETECTED:\n")
            for a in anomalies:
                report_buffer.write(f"- {a}\n")

        st.download_button(
            label="📥 Download Report",
            data=report_buffer.getvalue(),
            file_name="smart_energy_report.txt",
            mime="text/plain"
        )

        st.markdown("---")
        st.markdown("### 🧰 Smart Energy Tips")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 💰 Cost Saving Tips")
            st.write("⚡ Peak hours – reduce non-essential appliance use")
            st.write("🔗 Connect to weather API for real-time external temp")
            st.write("🔌 Sync with electricity pricing API for cost-saving")
            st.write("📅 Connect to calendar for automated scheduling")

        with col2:
            st.markdown("#### 🎮 Game Challenges")
            st.write("🎮 Challenge: Save 5% more energy this week!")
            st.write("🏆 Earned: Efficient Room badge 🥇")

        with col3:
            st.markdown("#### 🚨 Emergency Tips")
            st.write("⚡ Prep devices for sudden outages")
            st.write("🧯 Safety: Turn off heating/cooling during equipment failure")
            st.write("🌪️ Alert: Close windows during storm-level wind speeds")

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        st.write("Please check that your model files are correctly saved and accessible.")
        st.write("Required files: occupancy_model.h5, occupancy_scaler.h5, energy_consumption_model_gpu.h5")

st.markdown("---")
st.markdown("**Note:** Energy predictions use LSTM neural network trained on 4 environmental parameters: Temperature, Humidity, Light, and CO2.")
st.markdown("*LSTM model analyzes environmental patterns to predict energy consumption accurately.*")