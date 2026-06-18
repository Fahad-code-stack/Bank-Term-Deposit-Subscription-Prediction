import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config for a premium look
st.set_page_config(
    page_title="Bank Term Deposit Campaign Hub",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
        border: 1px solid #E5E7EB;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.75rem;
        font-size: 1.4rem;
        font-weight: 600;
        text-align: center;
        margin-top: 1rem;
    }
    .subscribe-bg {
        background-color: #DEF7EC;
        color: #03543F;
        border: 1px solid #BCF0DA;
    }
    .no-subscribe-bg {
        background-color: #FDE8E8;
        color: #9B1C1C;
        border: 1px solid #FBD5D5;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏦 Bank Term Deposit Prediction & Campaign Hub</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">An End-to-End Analytics, Reporting, and Machine Learning Prediction Platform</div>', unsafe_allow_html=True)

# Helper function to load model and encoders safely
@st.cache_resource
def load_assets():
    model_path = "bank_model.pkl"
    encoders_path = "encoders.pkl"
    dataset_path = "Bank_dataset.csv"
    
    # Check if files exist relative to script
    if not os.path.exists(model_path):
        model_path = os.path.join("ML LAB TRYING", "bank_model.pkl")
        encoders_path = os.path.join("ML LAB TRYING", "encoders.pkl")
        dataset_path = os.path.join("ML LAB TRYING", "Bank_dataset.csv")
        
    try:
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        return model, encoders, dataset_path
    except Exception as e:
        return None, None, None

model, encoders, dataset_path = load_assets()

# Sidebar Setup
with st.sidebar:
    st.header("Project Hub Navigation")
    st.markdown("Use this panel to learn about the system status.")
    if model is not None:
        st.success("🤖 Model loaded successfully!")
    else:
        st.error("❌ Model load failed!")
    
    st.info("""
    **Production Model Summary:**
    - **Classifier:** Random Forest
    - **Features Used:** 8 (Demographic & Campaign)
    - **Production Accuracy:** 87.76%
    """)
    st.markdown("---")
    st.subheader("Share & Deploy App")
    st.markdown("""
    Want to share this app with others?
    1. Push this folder to a GitHub Repository.
    2. Go to [share.streamlit.io](https://share.streamlit.io/).
    3. Connect your repository to deploy and get a public URL for free!
    """)

# Create Tabs
tab_predict, tab_dataset, tab_report = st.tabs([
    "🎯 Model Prediction", 
    "📊 Dataset Explorer", 
    "📄 Project Report"
])

# ================= TAB 1: PREDICTOR =================
with tab_predict:
    st.markdown("### Client Prediction Panel")
    st.markdown("Enter customer details below to calculate subscription probability.")

    if model is not None and encoders is not None:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input(
                    "Age", 
                    min_value=1, 
                    max_value=120, 
                    value=30, 
                    step=1,
                    help="Age of the customer"
                )
                
                job = st.selectbox(
                    "Job Profile", 
                    options=list(encoders["job"].classes_),
                    help="Type of job"
                )
                
                marital = st.selectbox(
                    "Marital Status", 
                    options=list(encoders["marital"].classes_),
                    help="Marital status of the customer"
                )
                
                education = st.selectbox(
                    "Education Level", 
                    options=list(encoders["education"].classes_),
                    help="Highest level of education"
                )

            with col2:
                balance = st.number_input(
                    "Average Annual Balance (in €)", 
                    value=1500.0, 
                    step=100.0,
                    help="Average annual balance of the client"
                )
                
                default = st.selectbox(
                    "Has Credit in Default?", 
                    options=list(encoders["default"].classes_),
                    help="Has credit in default status?"
                )
                
                housing = st.selectbox(
                    "Has Housing Loan?", 
                    options=list(encoders["housing"].classes_),
                    help="Has a housing loan?"
                )
                
                loan = st.selectbox(
                    "Has Personal Loan?", 
                    options=list(encoders["loan"].classes_),
                    help="Has a personal loan?"
                )

            submit_btn = st.form_submit_button("Generate Prediction Result", use_container_width=True)

        if submit_btn:
            # 1. Gather inputs
            sample = {
                "age": age,
                "job": job,
                "marital": marital,
                "education": education,
                "default": default,
                "balance": balance,
                "housing": housing,
                "loan": loan
            }
            
            # 2. Encode categorical features using loaded label encoders
            encoded_sample = sample.copy()
            try:
                for col in ["job", "marital", "education", "default", "housing", "loan"]:
                    if col in encoders:
                        encoded_sample[col] = encoders[col].transform([sample[col]])[0]
                
                # Ensure correct column order matching training features
                feature_order = ["age", "job", "marital", "education", "default", "balance", "housing", "loan"]
                sample_df = pd.DataFrame([encoded_sample])[feature_order]
                
                # Make prediction
                prediction = model.predict(sample_df)[0]
                probabilities = model.predict_proba(sample_df)[0]
                prob_subscribe = probabilities[1] * 100
                
                if prediction == 1:
                    st.markdown(
                        f'<div class="prediction-box subscribe-bg">🎉 Customer is likely to subscribe to the term deposit! <br><span style="font-size: 1.1rem; font-weight: normal;">Probability: {prob_subscribe:.2f}%</span></div>', 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="prediction-box no-subscribe-bg">❌ Customer is unlikely to subscribe to the term deposit. <br><span style="font-size: 1.1rem; font-weight: normal;">Probability of subscription: {prob_subscribe:.2f}%</span></div>', 
                        unsafe_allow_html=True
                    )
                    
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Model resources are not available.")

# ================= TAB 2: DATASET EXPLORER =================
with tab_dataset:
    st.markdown("### Dataset Statistics & Exploratory Data Analysis")
    
    if dataset_path and os.path.exists(dataset_path):
        @st.cache_data
        def load_df(path):
            return pd.read_csv(path)
        
        try:
            df = load_df(dataset_path)
            
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            with col_metric1:
                st.metric("Total Records", f"{df.shape[0]:,}")
            with col_metric2:
                st.metric("Total Features", f"{df.shape[1]}")
            with col_metric3:
                st.metric("Target Imbalance Ratio (y)", "88.7% No / 11.3% Yes")
            
            st.markdown("#### Sample Dataset View")
            st.dataframe(df.head(100), use_container_width=True)
            
            st.markdown("#### Feature Relationships and Distributions")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.write("**Age Distribution of Respondents**")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(data=df, x='age', hue='y', multiple='stack', bins=30, ax=ax, palette="Set2")
                ax.set_title("Age distribution grouped by Subscription Result (y)")
                st.pyplot(fig)
                
            with col_chart2:
                st.write("**Account Balance (€) Distribution**")
                fig, ax = plt.subplots(figsize=(6, 4))
                # Truncating display for outlier balance visualization safety
                sns.boxplot(data=df, x='y', y='balance', ax=ax, palette="Set2")
                ax.set_ylim(-2000, 10000) # clip outliers for better readability
                ax.set_title("Annual Balance by Subscription Result (y)")
                st.pyplot(fig)
                
        except Exception as e:
            st.warning(f"Could not load or plot dataset: {e}")
    else:
        st.info("Dataset file `Bank_dataset.csv` not found in current directory.")

# ================= TAB 3: REPORT =================
with tab_report:
    report_path = "report.md"
    if not os.path.exists(report_path):
        report_path = os.path.join("ML LAB TRYING", "report.md")
        
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(content)
    else:
        st.info("Report file `report.md` not found.")
