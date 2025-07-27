import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline

# Load your pre-trained pipeline
@st.cache_resource
def load_model():
    return joblib.load('loan_approval_pipeline.joblib')  # Update with your model path

pipeline = load_model()

# App title
st.title('Loan Approval Predictor')
st.write("""
Enter the applicant's details to predict loan approval status.
""")

# Input form
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        person_age = st.number_input('Age', min_value=18, max_value=100, value=30)
        person_gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
        person_education = st.selectbox('Education Level', 
                                      ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
        person_income = st.number_input('Annual Income ($)', min_value=0, value=50000)
        person_emp_exp = st.number_input('Employment Experience (years)', min_value=0.0, value=5.0)
        
    with col2:
        person_home_ownership = st.selectbox('Home Ownership', 
                                           ['Rent', 'Own', 'Mortgage', 'Other'])
        loan_amnt = st.number_input('Loan Amount ($)', min_value=100, value=10000)
        loan_intent = st.selectbox('Loan Purpose', 
                                 ['Debt Consolidation', 'Home Improvement', 'Business', 
                                  'Personal', 'Education', 'Medical'])
        loan_int_rate = st.number_input('Interest Rate (%)', min_value=0.0, max_value=30.0, value=10.0)
        loan_percent_income = st.slider('Loan Amount as % of Income', 0.01, 1.0, 0.2)
    
    cb_person_cred_hist_length = st.slider('Credit History Length (years)', 0, 30, 5)
    credit_score = st.slider('Credit Score', 300, 850, 650)
    previous_loan_defaults_on_file = st.selectbox('Previous Loan Defaults', ['No', 'Yes'])
    
    submitted = st.form_submit_button("Predict Approval")

# Prediction logic
if submitted:
    # Create input DataFrame
    input_data = pd.DataFrame({
        'person_age': [person_age],
        'person_gender': [person_gender],
        'person_education': [person_education],
        'person_income': [person_income],
        'person_emp_exp': [person_emp_exp],
        'person_home_ownership': [person_home_ownership],
        'loan_amnt': [loan_amnt],
        'loan_intent': [loan_intent],
        'loan_int_rate': [loan_int_rate],
        'loan_percent_income': [loan_percent_income],
        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
        'credit_score': [credit_score],
        'previous_loan_defaults_on_file': [previous_loan_defaults_on_file]
    })
    
    try:
        # Make prediction
        prediction = pipeline.predict(input_data)
        proba = pipeline.predict_proba(input_data)[0][1] * 100
        
        # Display results
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success(f'✅ Approved (Probability: {proba:.1f}%)')
        else:
            st.error(f'❌ Rejected (Probability: {proba:.1f}%)')
            
        # Show input summary
        with st.expander("See input details"):
            st.dataframe(input_data)
            
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Sidebar with info
st.sidebar.header("About")
st.sidebar.info("""
This app predicts loan approval based on applicant information.
The model uses machine learning to assess credit risk.
""")