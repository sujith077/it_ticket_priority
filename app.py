import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="IT Ticket Prioritization System", page_icon="🎫", layout="centered")

st.title("🎫 Smart IT Support Ticket Priority Predictor")
st.markdown("---")
st.write("This tool uses an optimized Random Forest Classifier to dynamically analyze and assign operational priorities to incoming IT support logs.")

@st.cache_resource
def load_ml_assets():
    model = joblib.load('priority_model.pkl')
    encoders = joblib.load('encoders.pkl')
    return model, encoders

try:
    model, encoders = load_ml_assets()
except:
    st.error("Error: Serialization deployment assets (`priority_model.pkl` or `encoders.pkl`) were not detected in the working environment pathway.")
    st.stop()

with st.form("prediction_form"):
    st.subheader("Ticket Feature Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        department = st.selectbox("Originating Department", encoders['Department'].classes_)
        category = st.selectbox("Functional Issue Category", encoders['Issue_Category'].classes_)
        device = st.selectbox("Primary Device Classification", encoders['Device_Type'].classes_)
    with col2:
        affected_users = st.slider("Scope of Impact (Affected Users)", min_value=1, max_value=50, value=1)
        business_critical = st.radio("Business Critical System Outage?", encoders['Business_Critical'].classes_, index=1)
        
    submit = st.form_submit_button("Compute System Priority Target")

if submit:
    input_df = pd.DataFrame([{
        'Department': encoders['Department'].transform([department])[0],
        'Issue_Category': encoders['Issue_Category'].transform([category])[0],
        'Device_Type': encoders['Device_Type'].transform([device])[0],
        'Affected_Users': affected_users,
        'Business_Critical': encoders['Business_Critical'].transform([business_critical])[0]
    }])
    
    prediction = model.predict(input_df)[0]
    
    design_map = {
        "Critical": {"emoji": "🔴", "color": "#ff4b4b"},
        "High": {"emoji": "orange_circle:", "color": "#ffa500"},
        "Medium": {"emoji": "🟡", "color": "#f1c40f"},
        "Low": {"emoji": "🟢", "color": "#2ecc71"}
    }
    
    meta = design_map.get(prediction, {"emoji": "ℹ️", "color": "#333333"})
    st.markdown(f"<div style='padding:20px; border-radius:10px; background-color:{meta['color']}; color:white; font-size:24px; font-weight:bold; text-align:center;'>Resulting Priority Level: {meta['emoji']} {prediction}</div>", unsafe_allowed_html=True)