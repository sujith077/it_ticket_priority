import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="IT Ticket Prioritization System", page_icon="🎫", layout="centered")

# --- USER LOGIN GATEWAY ---
# Force login check before allowing form access
if not st.user.is_logged_in:
    st.title("🎫 Secure IT Support Gateway")
    st.info("Please log in using your corporate account credentials to submit a support ticket.")
    st.button("Log in with Company Email", on_click=st.login)
    st.stop()  # Terminate script here if unauthenticated

# Extract user profile email securely after login
user_email = st.user.email

st.title("🎫 Smart IT Support Ticket Priority Predictor")
st.sidebar.markdown(f"**Logged in as:**\n`{user_email}`")
if st.sidebar.button("Log Out"):
    st.logout()

st.write("This tool uses an optimized Random Forest Classifier to assign operational response priorities to support tickets.")
st.markdown("---")

@st.cache_resource
def load_ml_assets():
    model = joblib.load('priority_model.pkl')
    encoders = joblib.load('encoders.pkl')
    return model, encoders

try:
    model, encoders = load_ml_assets()
except:
    st.error("Error: Serialized deployment assets were not detected in the working environment pathway.")
    st.stop()

with st.form("prediction_form"):
    st.subheader("New Ticket Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        department = st.selectbox("Originating Department", encoders['Department'].classes_)
        category = st.selectbox("Functional Issue Category", encoders['Issue_Category'].classes_)
        device = st.selectbox("Primary Device Classification", encoders['Device_Type'].classes_)
    with col2:
        # User explicitly selects their branch location
        branch = st.selectbox("Office Branch Location", encoders['Office_Location'].classes_)
        affected_users = st.slider("Scope of Impact (Affected Users)", min_value=1, max_value=50, value=1)
        business_critical = st.radio("Business Critical System Outage?", encoders['Business_Critical'].classes_, index=1)
        
    submit = st.form_submit_button("Compute System Priority Target")

if submit:
    # Build feature row exactly matching training shapes
    input_df = pd.DataFrame([{
        'Department': encoders['Department'].transform([department])[0],
        'Issue_Category': encoders['Issue_Category'].transform([category])[0],
        'Device_Type': encoders['Device_Type'].transform([device])[0],
        'Affected_Users': affected_users,
        'Business_Critical': encoders['Business_Critical'].transform([business_critical])[0],
        'Office_Location': encoders['Office_Location'].transform([branch])[0]
    }])
    
    prediction = model.predict(input_df)[0]
    
    design_map = {
        "Critical": {"emoji": "🔴", "color": "#ff4b4b"},
        "High": {"emoji": "🟠", "color": "#ffa500"},
        "Medium": {"emoji": "🟡", "color": "#f1c40f"},
        "Low": {"emoji": "🟢", "color": "#2ecc71"}
    }
    
    meta = design_map.get(prediction, {"emoji": "ℹ️", "color": "#333333"})
    
    st.markdown(f"<div style='padding:20px; border-radius:10px; background-color:{meta['color']}; color:white; font-size:24px; font-weight:bold; text-align:center;'>Resulting Priority Level: {meta['emoji']} {prediction}</div>", unsafe_allow_html=True)
    
    # Audit log simulation showing who submitted the ticket
    st.info(f"Ticket logged under identity: **{user_email}** at **{branch}** Branch.")
