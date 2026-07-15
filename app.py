import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# Set page configuration
st.set_page_config(
    page_title="IT Ticket Priority Predictor",
    page_icon="🎫",
    layout="centered"
)

# --- 1. LOAD MODEL & ENCODERS ---
@st.cache_resource
def load_assets():
    # Load the optimized 7-feature model and label encoders
    model = joblib.load('priority_model.pkl')
    encoders = joblib.load('encoders.pkl')
    return model, encoders

try:
    model, encoders = load_assets()
except Exception as e:
    st.error(f"⚠️ Error loading model assets. Make sure 'priority_model.pkl' and 'encoders.pkl' are in your repository root. Error: {e}")
    st.stop()

# --- 2. APP HEADER ---
st.title("🎫 IT Ticket Priority Predictor")
st.markdown("""
This system uses a **Hybrid Validation Engine** to classify incoming support tickets. 
It combines a programmatic safety-net override for critical hardware infrastructure with a trained **Random Forest Classifier** for standard tickets.
""")

st.write("---")

# --- 3. INPUT FORM ---
st.subheader("Submit New Support Ticket")

with st.form("ticket_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        department = st.selectbox(
            "Requesting Department",
            options=["HR", "Finance", "IT", "Sales", "Marketing", "Operations"]
        )
        
        category = st.selectbox(
            "Issue Category",
            options=["Software", "Hardware", "Network", "Access/Login", "Security"]
        )
        
        device = st.selectbox(
            "Device Type",
            options=["Laptop", "Desktop", "Mobile", "Printer", "Server", "None"]
        )

    with col2:
        branch = st.selectbox(
            "Office Location",
            options=["Colombo", "Kandy"]
        )
        
        affected_users = st.number_input(
            "Number of Affected Users",
            min_value=1,
            max_value=1000,
            value=1,
            step=1
        )
        
        impact_choice = st.radio(
            "Business Operational Impact",
            options=["I can still work (Non-blocking)", "I cannot work (Blocker)"]
        )

    submit = st.form_submit_button("Classify Ticket Priority", type="primary")

# --- 4. PREDICTION WORKFLOW & HYBRID VALIDATION ---
if submit:
    # 1. Field validation check
    if not department or not category or not device or not branch or not impact_choice:
        st.error("⚠️ Submission Rejected: Please select a value for all fields.")
    else:
        # Pre-process binary status for the model pipeline
        business_critical = "Yes" if "I cannot work" in impact_choice else "No"
        
        # --- PATHWAY A: HYBRID LOGIC / PROGRAMMATIC SAFETY OVERRIDE ---
        if category == "Security" and device == "Server":
            if affected_users > 10 or business_critical == "Yes":
                prediction = "Critical"
            else:
                prediction = "High"
            
            st.warning(f"⚠️ **Safety Net Triggered:** Core Server Infrastructure security threat detected.")
            
        # --- PATHWAY B: OPTIMIZED ML INFERENCE PATHWAY (7 FEATURES) ---
        else:
            # Calculate the feature engineered term on-the-fly to prevent shape mismatch
            impact_multiplier = 1.5 if business_critical == "Yes" else 0.5
            calculated_impact_score = affected_users * impact_multiplier

            # Construct DataFrame with transformed categories
            input_df = pd.DataFrame([{
                'Department': encoders['Department'].transform([department])[0],
                'Issue_Category': encoders['Issue_Category'].transform([category])[0],
                'Device_Type': encoders['Device_Type'].transform([device])[0],
                'Affected_Users': affected_users,
                'Business_Critical': encoders['Business_Critical'].transform([business_critical])[0],
                'Office_Location': encoders['Office_Location'].transform([branch])[0],
                'Impact_Score': calculated_impact_score
            }])
            
            # Align features with the model training dataset's exact column order
            column_order = [
                'Department', 
                'Issue_Category', 
                'Device_Type', 
                'Affected_Users', 
                'Business_Critical', 
                'Office_Location', 
                'Impact_Score'
            ]
            input_df = input_df[column_order]
            
            # Run model prediction
            prediction = model.predict(input_df)[0]

        # --- 5. DISPLAY RESULTS ---
        st.success("### Ticket Processing Complete")
        
        # Color coordinate the UI feedback badge based on severity
        colors = {
            "Critical": "red",
            "High": "orange",
            "Medium": "blue",
            "Low": "green"
        }
        color = colors.get(prediction, "grey")
        
        st.markdown(f"Calculated Ticket Priority: :{color}[**{prediction}**]")
        
        # Optional: Append log output to JSON database locally
        ticket_log = {
            "Department": department,
            "Issue_Category": category,
            "Device_Type": device,
            "Affected_Users": affected_users,
            "Business_Critical": business_critical,
            "Office_Location": branch,
            "Assigned_Priority": prediction
        }
        
        try:
            log_file = "tickets.json"
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(ticket_log)
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=4)
        except Exception:
            pass # Silent fail if local JSON persistence is restricted
