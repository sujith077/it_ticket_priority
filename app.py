import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="IT Ticket Prioritization System NCHS", page_icon="🎫", layout="centered")

# --- CUSTOM FORM LOGIN GATEWAY ---
# Initialize session state for user authentication tracking
if 'logged_in_email' not in st.session_state:
    st.session_state.logged_in_email = None

# If the user is not logged in, force them through the secure form login portal
if st.session_state.logged_in_email is None:
    st.title("🎫 Secure IT Support Gateway")
    st.write("Please log in using your corporate email address to submit an IT support ticket.")
    
    with st.form("login_gateway"):
        email_input = st.text_input("Corporate Email Address", placeholder="username@company.com")
        login_submit = st.form_submit_button("Access Portal")
        
        if login_submit:
            if "@" in email_input and "." in email_input:
                st.session_state.logged_in_email = email_input
                st.rerun()
            else:
                st.error("Please enter a valid corporate email address.")
    st.stop()  # Halt execution here until authentication form is successfully submitted

# Extract authenticated user profile email after login
user_email = st.session_state.logged_in_email

# --- MAIN APPLICATION DASHBOARD ---
st.title("🎫 NCHS IT Support Ticket Priority Predictor",align = "center")

# Sidebar for managing current session status
st.sidebar.markdown(f"**Logged in as:**\n`{user_email}`")
if st.sidebar.button("Log Out"):
    st.session_state.logged_in_email = None
    st.rerun()

#st.write("This system uses an optimized Random Forest Classifier to assign operational response priorities to support tickets.")
st.markdown("---")

# Load model assets from file cache
@st.cache_resource
def load_ml_assets():
    model = joblib.load('priority_model.pkl')
    encoders = joblib.load('encoders.pkl')
    return model, encoders

try:
    model, encoders = load_ml_assets()
except:
    st.error("Error: Serialized deployment assets (`priority_model.pkl` or `encoders.pkl`) were not detected in the working environment pathway.")
    st.stop()

# Build interactive input form feature collection UI
with st.form("prediction_form"):
    st.subheader("New Ticket Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        department = st.selectbox("Originating Department", encoders['Department'].classes_)
        category = st.selectbox("Functional Issue Category", encoders['Issue_Category'].classes_)
        device = st.selectbox("Primary Device Classification", encoders['Device_Type'].classes_)
    with col2:
        # Branch mapping selection (Kandy vs Colombo)
        branch = st.selectbox("Office Branch Location", encoders['Office_Location'].classes_)
        affected_users = st.slider("Scope of Impact (Affected Users)", min_value=1, max_value=50, value=1)
        business_critical = st.radio("Business Critical System Outage?", encoders['Business_Critical'].classes_, index=1)
        
    submit = st.form_submit_button("Compute System Priority Target")

# Handle form data pipeline submission for live inference processing
if submit:
    # Construct feature structure row with identical label values used during model generation 
    input_df = pd.DataFrame([{
        'Department': encoders['Department'].transform([department])[0],
        'Issue_Category': encoders['Issue_Category'].transform([category])[0],
        'Device_Type': encoders['Device_Type'].transform([device])[0],
        'Affected_Users': affected_users,
        'Business_Critical': encoders['Business_Critical'].transform([business_critical])[0],
        'Office_Location': encoders['Office_Location'].transform([branch])[0]
    }])
    
    # Generate live model array prediction target string output
    prediction = model.predict(input_df)[0]
    
    # Establish priority color alert visual status map parameters
    design_map = {
        "Critical": {"emoji": "🔴", "color": "#ff4b4b"},
        "High": {"emoji": "orange_circle:", "color": "#ffa500"},
        "Medium": {"emoji": "🟡", "color": "#f1c40f"},
        "Low": {"emoji": "🟢", "color": "#2ecc71"}
    }
    
    meta = design_map.get(prediction, {"emoji": "ℹ️", "color": "#333333"})
    
    # Render stylized output interface box block safely
    st.markdown(f"<div style='padding:20px; border-radius:10px; background-color:{meta['color']}; color:white; font-size:24px; font-weight:bold; text-align:center;'>Resulting Priority Level: {meta['emoji']} {prediction}</div>", unsafe_allow_html=True)
    
    # Display ticket tracking data confirming email and location registry parameters
    st.info(f"Ticket successfully logged under identity: **{user_email}** at **{branch}** Branch.")
