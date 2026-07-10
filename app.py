import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="NCHS IT Ticket System", page_icon="🎫", layout="centered")

# --- DATABASE INITIALIZATION ---
# Initialize persistent session storage for the admin's master ticket registry
if 'ticket_database' not in st.session_state:
    st.session_state.ticket_database = []

# --- CUSTOM FORM LOGIN GATEWAY ---
if 'logged_in_email' not in st.session_state:
    st.session_state.logged_in_email = None

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
    st.stop()

user_email = st.session_state.logged_in_email

# --- MAIN APPLICATION INTERFACE ---
st.markdown("<h1 style='text-align: center;'>NCHS IT Support Ticket Priority Predictor</h1>", unsafe_allow_html=True)

st.sidebar.markdown(f"**Logged in as:**\n`{user_email}`")
if st.sidebar.button("Log Out"):
    st.session_state.logged_in_email = None
    st.rerun()

#st.write("Submit your technical issue below. Our machine learning system will automatically categorize and route your request based on operational urgency.")
st.markdown("---")

@st.cache_resource
def load_ml_assets():
    model = joblib.load('priority_model.pkl')
    encoders = joblib.load('encoders.pkl')
    return model, encoders

try:
    model, encoders = load_ml_assets()
except:
    st.error("Error: Serialized deployment assets (`priority_model.pkl` or `encoders.pkl`) were not detected.")
    st.stop()

# --- TICKET SUBMISSION FORM ---
with st.form("prediction_form"):
    st.subheader("New Ticket Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        department = st.selectbox("Originating Department", encoders['Department'].classes_)
        category = st.selectbox("Functional Issue Category", encoders['Issue_Category'].classes_)
        device = st.selectbox("Primary Device Classification", encoders['Device_Type'].classes_)
    with col2:
        branch = st.selectbox("Office Branch Location", encoders['Office_Location'].classes_)
        affected_users = st.slider("Scope of Impact (Affected Users)", min_value=1, max_value=50, value=1)
        
        # Non-technical literacy option for user convenience
        impact_choice = st.radio(
            "How badly is this issue affecting your work?",
            [
                "🔴 I cannot work at all (Entire system or department is down)", 
                "🟢 I can still work (Annoying problem, but work can continue)"
            ]
        )
        business_critical = "Yes" if "I cannot work at all" in impact_choice else "No"
        
    st.write("") # Spacing element
    
    # Split layout into columns to balance and perfectly center the submit button
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        submit = st.form_submit_button("Compute System Priority Target", use_container_width=True)

if submit:
    # Build data row format matching your Google Colab machine learning pipeline shape
    input_df = pd.DataFrame([{
        'Department': encoders['Department'].transform([department])[0],
        'Issue_Category': encoders['Issue_Category'].transform([category])[0],
        'Device_Type': encoders['Device_Type'].transform([device])[0],
        'Affected_Users': affected_users,
        'Business_Critical': encoders['Business_Critical'].transform([business_critical])[0],
        'Office_Location': encoders['Office_Location'].transform([branch])[0]
    }])
    
    # Run dynamic classification inference
    prediction = model.predict(input_df)[0]
    
    design_map = {
        "Critical": {"emoji": "🔴", "color": "#ff4b4b"},
        "High": {"emoji": "🟠", "color": "#ffa500"},
        "Medium": {"emoji": "🟡", "color": "#f1c40f"},
        "Low": {"emoji": "🟢", "color": "#2ecc71"}
    }
    meta = design_map.get(prediction, {"emoji": "ℹ️", "color": "#333333"})
    
    # Render stylized priority notification block banner
    st.markdown(f"<div style='padding:20px; border-radius:10px; background-color:{meta['color']}; color:white; font-size:24px; font-weight:bold; text-align:center;'>Resulting Priority Level: {meta['emoji']} {prediction}</div>", unsafe_allow_html=True)
    st.info(f"Ticket successfully logged under identity: **{user_email}** at **{branch}** Branch.")

    # Save transaction history entry to runtime queue matrix array
    new_ticket_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "User_Email": user_email,
        "Branch_Location": branch,
        "Department": department,
        "Issue_Category": category,
        "Device_Type": device,
        "Affected_Users": affected_users,
        "Business_Critical": business_critical,
        "Assigned_Priority": prediction
    }
    st.session_state.ticket_database.append(new_ticket_entry)
    
    # Send a quick toast confirmation alert across screen
    st.toast(f"Notification: New {prediction} priority ticket added to Admin View!", icon="🚨")

# --- ADMIN PANEL SECTION ---
st.markdown("---")
st.subheader("👨‍💻 System Administrator Control Portal")

if len(st.session_state.ticket_database) > 0:
    st.success("🔔 Notification: There are active tickets awaiting review in the queue below.")
    
    admin_df = pd.DataFrame(st.session_state.ticket_database)
    st.dataframe(admin_df, use_container_width=True)
    
    csv_download_data = admin_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Master Ticket Prioritization List (CSV)",
        data=csv_download_data,
        file_name="master_it_prioritization_list.csv",
        mime="text/csv"
    )
else:
    st.info("No active tickets logged in the current application session runtime environment queue.")
