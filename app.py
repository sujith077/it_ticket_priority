import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="NCHS IT Ticket System", page_icon="🎫", layout="centered")

# --- DATABASE INITIALIZATION ---
if 'ticket_database' not in st.session_state:
    st.session_state.ticket_database = []

# --- CUSTOM FORM LOGIN GATEWAY ---
if 'logged_in_user' not in st.session_state:
    st.session_state.logged_in_user = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = None

if st.session_state.logged_in_user is None:
    st.title("🎫 Secure IT Support Gateway")
    st.write("Please log in to access the portal.")
    
    with st.form("login_gateway"):
        username_input = st.text_input("Username / Email Address", placeholder="username@nchs.edu.lk")
        password_input = st.text_input("Password", type="password", placeholder="••••••••")
        login_submit = st.form_submit_button("Access Portal")
        
        if login_submit:
            clean_user = username_input.strip().lower()
            
            if clean_user == "itsupport@nchs.edu.lk" and password_input == "admin@123":
                st.session_state.logged_in_user = clean_user
                st.session_state.user_role = "Admin"
                st.rerun()
            elif "@" in clean_user and "." in clean_user:
                st.session_state.logged_in_user = clean_user
                st.session_state.user_role = "User"
                st.rerun()
            else:
                st.error("Invalid credentials. Please enter a valid corporate email profile or correct admin passwords.")
    st.stop()

user_email = st.session_state.logged_in_user
is_admin = (st.session_state.user_role == "Admin")

# --- MAIN APPLICATION INTERFACE ---
if is_admin:
    st.markdown("<h1 style='text-align: center;'>👨‍💻 NCHS IT System Administrator Portal</h1>", unsafe_allow_html=True)
else:
    st.markdown("<h1 style='text-align: center;'>🎫 NCHS IT Support Ticket Priority Predictor</h1>", unsafe_allow_html=True)

st.sidebar.markdown(f"**Logged in as:**\n`{user_email}`")
st.sidebar.markdown(f"**Role:** {st.session_state.user_role}")
if st.sidebar.button("Log Out"):
    st.session_state.logged_in_user = None
    st.session_state.user_role = None
    st.rerun()

# ----------------- ADMIN INTERFACE -----------------
if is_admin:
    st.write("Welcome to the control console. Below is the live queue of raised tickets and their predicted classifications.")
    st.markdown("---")
    
    if len(st.session_state.ticket_database) > 0:
        admin_df = pd.DataFrame(st.session_state.ticket_database)
        
        # --- NEW METRICS & VISUALIZATIONS SECTION ---
        st.subheader("📊 System Performance Metrics & Charts")
        
        # Key Performance Indicator (KPI) Cards
        kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
        with kpi_col1:
            st.metric(label="Total Tickets Logged", value=len(admin_df))
        with kpi_col2:
            critical_count = len(admin_df[admin_df['Assigned_Priority'] == 'Critical'])
            st.metric(label="🚨 Critical Escalations", value=critical_count)
        with kpi_col3:
            avg_impact = round(admin_df['Affected_Users'].mean(), 1)
            st.metric(label="👥 Avg. Impact Radius", value=f"{avg_impact} Users")
            
        # Graphical Charts Layout
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("**Tickets Grouped by Priority**")
            priority_counts = admin_df['Assigned_Priority'].value_counts()
            # Ensure proper ordering of priority levels if present
            order = ['Critical', 'High', 'Medium', 'Low']
            priority_counts = priority_counts.reindex([p for p in order if p in priority_counts.index])
            st.bar_chart(priority_counts)
            
        with chart_col2:
            st.markdown("**Ticket Breakdown by Branch Location**")
            branch_counts = admin_df['Branch_Location'].value_counts()
            st.bar_chart(branch_counts)
            
        st.markdown("**📈 Incident Submission Volume Timeline**")
        timeline_df = admin_df.groupby('Timestamp').size()
        st.line_chart(timeline_df)
        
        st.markdown("---")
        st.subheader("📋 Active Operations Data Queue")
        
        st.dataframe(admin_df, use_container_width=True)
        
        csv_download_data = admin_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Master Ticket Prioritization List (CSV)",
            data=csv_download_data,
            file_name="master_it_prioritization_list.csv",
            mime="text/csv"
        )
    else:
        st.info("No active tickets logged in the current session queue to generate analytical metrics.")

# ----------------- USER INTERFACE -----------------
else:
    st.write("Submit your technical issue below. Our machine learning system will automatically categorize and route your request based on operational urgency.")
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

    with st.form("prediction_form"):
        st.subheader("New Ticket Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            department = st.selectbox("Originating Department", encoders['Department'].classes_, index=None, placeholder="Choose Department...")
            category = st.selectbox("Functional Issue Category", encoders['Issue_Category'].classes_, index=None, placeholder="Choose Category...")
            device = st.selectbox("Primary Device Classification", encoders['Device_Type'].classes_, index=None, placeholder="Choose Device...")
        with col2:
            branch = st.selectbox("Office Branch Location", encoders['Office_Location'].classes_, index=None, placeholder="Choose Branch...")
            affected_users = st.slider("Scope of Impact (Affected Users)", min_value=1, max_value=50, value=1)
            
            impact_choice = st.radio(
                "How badly is this issue affecting your work?",
                ["🔴 I cannot work", "🟢 I can still work"],
                index=None
            )
            
        st.write("") 
        
        btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
        with btn_col2:
            submit = st.form_submit_button("Compute System Priority Target", use_container_width=True)

    if submit:
        if not department or not category or not device or not branch or not impact_choice:
            st.error("⚠️ Submission Rejected: Please select a value for all fields before computing priority.")
        else:
            business_critical = "Yes" if "I cannot work" in impact_choice else "No"
            
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
            st.info(f"Ticket successfully logged under identity: **{user_email}** at **{branch}** Branch.")

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
            st.toast(f"Ticket successfully logged!", icon="✅")
