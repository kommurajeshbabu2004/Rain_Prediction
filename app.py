import streamlit as st
import pandas as pd
import joblib
import datetime

# Load the trained classification model
model = joblib.load('rainfall_model.pkl.xz')

# Initialize a session state to store the history
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=["conds", "hum", "pressurem", "vism", "wdird", "wdire", "wspdm", "hour", "day", "month", "weekday", "dew_point", "wind_chill", "prediction"])

st.title("🌧 Rain Prediction App")

st.write("Enter weather details in the sidebar and click *Predict*.")

# Sidebar inputs
st.sidebar.header("🌦 Enter Weather Data")

# Weather Condition Selection
conds = st.sidebar.selectbox("Weather Condition", ["Clear", "Cloudy", "Rain", "Thunderstorm"])
conds_mapping = {"Clear": 0.0, "Cloudy": 1.0, "Rain": 2.0, "Thunderstorm": 3.0}
conds = conds_mapping[conds]

# Other Weather Inputs
hum = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
pressurem = st.sidebar.number_input("Pressure (mbar)", 900.0, 1100.0, 1013.0)
vism = st.sidebar.number_input("Visibility (km)", 0.0, 20.0, 10.0)
wdird = st.sidebar.slider("Wind Direction (°)", 0.0, 360.0, 180.0)

# Wind Direction as Text (Convert to Numeric)
wdire = st.sidebar.selectbox("Wind Direction (Text)", ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
wdire_mapping = {"N": 0.0, "NE": 1.0, "E": 2.0, "SE": 3.0, "S": 4.0, "SW": 5.0, "W": 6.0, "NW": 7.0}
wdire = wdire_mapping[wdire]

wspdm = st.sidebar.slider("Wind Speed (km/h)", 0.0, 100.0, 10.0)
dew_point = st.sidebar.number_input("Dew Point (°C)", -10.0, 30.0, 15.0)
wind_chill = st.sidebar.number_input("Wind Chill (°C)", -10.0, 30.0, 15.0)

# Automatically capture datetime features
current_time = datetime.datetime.now()
hour = float(current_time.hour)
day = float(current_time.day)
month = float(current_time.month)
weekday = float(current_time.weekday())

# Prepare input data with all required columns
input_data = pd.DataFrame([[conds, hum, pressurem, vism, wdird, wdire, wspdm, hour, day, month, weekday, dew_point, wind_chill]],
                          columns=["conds", "hum", "pressurem", "vism", "wdird", "wdire", "wspdm", "hour", "day", "month", "weekday", "dew_point", "wind_chill"])

# Display user inputs for debugging
st.write("🔍 *Input Data for Prediction:*")
st.write(input_data)

# Predict Rainfall (Classification: 0 = No Rain, 1 = Rain)
if st.sidebar.button("🚀 Predict Rain"):
    prediction = model.predict(input_data)
    rain_status = "🌞 No Rain" if prediction[0] == 0 else "🌧 Rain Expected"
    
    # Add prediction to the input data
    input_data["prediction"] = rain_status
    
    # Append the prediction history to session state
    st.session_state.history = pd.concat([st.session_state.history, input_data], ignore_index=True)
    
    # Display prediction result
    st.subheader(f"*Prediction: {rain_status}*")

# Display history of predictions
st.write("📜 Prediction History:")
st.write(st.session_state.history)

# Allow users to download the history as CSV
csv = st.session_state.history.to_csv(index=False)
st.download_button(
    label="Download Prediction History (CSV)",
    data=csv,
    file_name="prediction_history.csv",
    mime="text/csv"
)
