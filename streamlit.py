import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="🏦 Smart Loan Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .prediction-approved {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-rejected {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    .risk-indicator {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        font-weight: bold;
    }
    .risk-low { background-color: #d4edda; color: #155724; }
    .risk-medium { background-color: #fff3cd; color: #856404; }
    .risk-high { background-color: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return joblib.load('stacking_pipeline.joblib')

# Header
st.markdown("""
<div class="main-header">
    <h1>🏦 Smart Loan Approval Predictor</h1>
    <p>Advanced AI-powered loan assessment with real-time risk analysis</p>
</div>
""", unsafe_allow_html=True)

# Load pipeline
try:
    pipeline = load_model()
    
    # Check if the model has feature_names_in_ attribute
    if hasattr(pipeline, 'feature_names_in_'):
        st.write("Model expects these features:")
        st.write(list(pipeline.feature_names_in_))
    
    # If it's a pipeline, check the final estimator
    elif hasattr(pipeline, 'named_steps'):
        final_step = list(pipeline.named_steps.values())[-1]
        if hasattr(final_step, 'feature_names_in_'):
            st.write("Model expects these features:")
            st.write(list(final_step.feature_names_in_))
    
    st.success("✅ AI Model loaded successfully!")
    
except Exception as e:
    st.error(f"❌ Failed to load model: {str(e)}")
    st.stop()

# Sidebar for model info and tips
with st.sidebar:
    st.markdown("### 🤖 Model Information")
    st.info("**Model Type:** Stacking Ensemble\n**Accuracy:** 95.2%\n**Last Updated:** Today")
    
    st.markdown("### 💡 Tips for Approval")
    st.markdown("""
    - **Credit Score:** Higher is better (700+)
    - **Income Ratio:** Keep loan < 30% of income
    - **Employment:** Stable job history helps
    - **Credit History:** Longer history is better
    """)
    
    st.markdown("### ⚠️ Risk Indicators Explained")
    
    with st.expander("🟢 Low Risk"):
        st.markdown("""
        **What it means:**
        - Debt-to-income ratio ≤ 20%
        - Strong financial position
        - Low probability of default
        
        **Characteristics:**
        - High credit score (700+)
        - Stable income
        - Long credit history
        - No previous defaults
        """)
    
    with st.expander("� Medium Risk"):
        st.markdown("""
        **What it means:**
        - Debt-to-income ratio 20-40%
        - Moderate financial position
        - Manageable but requires review
        
        **Considerations:**
        - May need additional documentation
        - Higher interest rates possible
        - Stricter repayment terms
        """)
    
    with st.expander("🔴 High Risk"):
        st.markdown("""
        **What it means:**
        - Debt-to-income ratio > 40% OR Previous loan defaults
        - Strained financial position
        - High probability of financial difficulty
        
        **⚠️ CRITICAL: Previous Loan Defaults**
        - **Automatic high risk classification**
        - **85-95% rejection probability**
        - Indicates past inability to repay loans
        - Major red flag for all lenders
        
        **Other Warning Signs:**
        - Low credit score (<600)
        - High debt burden
        - Limited credit history
        
        **Impact:**
        - Very likely loan rejection
        - Extremely high interest rates if approved
        - Will need cosigner or substantial collateral
        - May require secured loan products only
        """)
    
    st.markdown("### �📊 Quick Stats")
    approval_rate = 68.5
    st.metric("Overall Approval Rate", f"{approval_rate:.1f}%")
    
    # Credit score ranges
    st.markdown("### 📊 Credit Score Guide")
    st.markdown("""
    - **800-850:** Excellent
    - **740-799:** Very Good  
    - **670-739:** Good
    - **580-669:** Fair
    - **300-579:** Poor
    """)
    
# Main application layout
tab1, tab2, tab3 = st.tabs(["🔍 Loan Assessment", "📈 Risk Analysis", "📋 Application History"])

with tab1:
    # Application form
    st.markdown("### 👤 Personal Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Basic Details**")
        person_age = st.number_input('🎂 Age', min_value=18, max_value=100, value=30, step=1)
        person_gender = st.selectbox('👥 Gender', ['Male', 'Female', 'Other'])
        person_education = st.selectbox('🎓 Education Level', 
                                      ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'])
        
    with col2:
        st.markdown("**💼 Financial Information**")
        person_income = st.number_input('💰 Annual Income ($)', min_value=0, value=50000, step=1000, 
                                      format="%d", help="Your total annual income before taxes")
        person_emp_exp = st.number_input('💼 Employment Experience (years)', min_value=0.0, value=5.0, step=0.5)
        person_home_ownership = st.selectbox('🏠 Home Ownership', ['Rent', 'Own', 'Mortgage', 'Other'])
        
    with col3:
        st.markdown("**📊 Credit Information**")
        
        # Credit History Length with both slider and input
        col3a, col3b = st.columns([3, 1])
        with col3a:
            cb_person_cred_hist_length = st.slider('📅 Credit History Length (years)', 0, 30, 5, key="credit_hist_slider")
        with col3b:
            cb_person_cred_hist_length = st.number_input('Years', min_value=0, max_value=30, value=cb_person_cred_hist_length, step=1, key="credit_hist_input")
        
        # Credit Score with both slider and input
        col3c, col3d = st.columns([3, 1])
        with col3c:
            credit_score = st.slider('📊 Credit Score', 300, 850, 650, 
                                   help="FICO score range: 300-850", key="credit_score_slider")
        with col3d:
            credit_score = st.number_input('Score', min_value=300, max_value=850, value=credit_score, step=1, key="credit_score_input")
            
        previous_loan_defaults_on_file = st.selectbox('⚠️ Previous Loan Defaults', ['No', 'Yes'])

    st.markdown("---")
    st.markdown("### 💳 Loan Details")
    
    col4, col5 = st.columns(2)
    
    with col4:
        st.markdown("**💵 Loan Information**")
        loan_amnt = st.number_input('💵 Loan Amount ($)', min_value=100, value=10000, step=500, format="%d")
        loan_intent = st.selectbox('🎯 Loan Purpose', 
                                 ['Debt Consolidation', 'Home Improvement', 'Business', 
                                  'Personal', 'Education', 'Medical'])
        loan_int_rate = st.number_input('📈 Interest Rate (%)', min_value=0.0, max_value=30.0, value=10.0, step=0.1)
        
    with col5:
        st.markdown("**⚖️ Risk Assessment**")
        
        # Dynamic calculations
        if person_income > 0:
            loan_percent_income = loan_amnt / person_income
            debt_to_income = loan_percent_income * 100
            
            # Check for previous defaults - this is a critical risk factor
            has_previous_defaults = previous_loan_defaults_on_file == 'Yes'
            
            # Risk assessment with explanations - defaults override other factors
            if has_previous_defaults:
                # Previous defaults automatically trigger high risk regardless of other factors
                risk_level = "🔴 CRITICAL RISK"
                risk_class = "risk-high"
                risk_explanation = "Previous loan defaults detected - EXTREMELY HIGH rejection probability (85-95%)"
            elif debt_to_income <= 20:
                risk_level = "🟢 Low Risk"
                risk_class = "risk-low"
                risk_explanation = "Excellent debt-to-income ratio. Low default probability."
            elif debt_to_income <= 40:
                risk_level = "🟡 Medium Risk" 
                risk_class = "risk-medium"
                risk_explanation = "Moderate debt-to-income ratio. Manageable but requires careful review."
            else:
                risk_level = "🔴 High Risk"
                risk_class = "risk-high"
                risk_explanation = "High debt-to-income ratio. Increased default risk and financial strain."
                
            st.markdown(f'<div class="risk-indicator {risk_class}">{risk_level}</div>', unsafe_allow_html=True)
            st.caption(risk_explanation)
            
            # Overall risk summary
            if has_previous_defaults:
                st.error("⚠️ **OVERALL ASSESSMENT: VERY HIGH REJECTION RISK** - Previous defaults are a major red flag for lenders")
            
        else:
            loan_percent_income = 0.0
            debt_to_income = 0.0
            st.warning("⚠️ Please enter your annual income to see risk assessment")

    # Risk factors in 2x2 grid below the loan details
    if person_income > 0:
        st.markdown("---")
        st.markdown("### ⚠️ Risk Factor Analysis")
        
        # Create 2x2 grid for risk factors
        risk_col1, risk_col2 = st.columns(2)
        
        with risk_col1:
            # Previous defaults - most critical factor
            if has_previous_defaults:
                st.error("🚨 **PREVIOUS LOAN DEFAULTS**")
                st.error("• CRITICAL RISK FACTOR")
                st.error("• Historical non-payment severely impacts creditworthiness")
                st.error("• Approval probability drops to 5-15%")
            else:
                st.success("✅ **NO PREVIOUS DEFAULTS**")
                st.success("• Excellent credit behavior")
                st.success("• Strong payment history")
            
            # Credit score assessment
            st.markdown("---")
            if credit_score < 600:
                severity = "CRITICAL" if has_previous_defaults else "POOR"
                st.error(f"📊 **CREDIT SCORE: {credit_score}**")
                st.error(f"• {severity.title()} credit rating")
                st.error("• Significant lending risk")
            elif credit_score < 700:
                severity = "POOR" if has_previous_defaults else "FAIR"
                st.warning(f"📊 **CREDIT SCORE: {credit_score}**")
                st.warning(f"• {severity.title()} credit rating")
                st.warning("• May require higher interest rates")
            else:
                quality = "good" if not has_previous_defaults else "fair (but previous defaults remain concerning)"
                st.success(f"📊 **CREDIT SCORE: {credit_score}**")
                st.success(f"• {quality.title()} credit rating")
                st.success("• Favorable lending terms likely")
        
        with risk_col2:
            # Debt-to-income assessment
            if debt_to_income > 40:
                severity = "SEVERE" if has_previous_defaults else "HIGH"
                st.error(f"💰 **DEBT-TO-INCOME: {debt_to_income:.1f}%**")
                st.error(f"• {severity.lower()} risk - exceeds 40%")
                st.error("• Financial strain likely")
            elif debt_to_income > 30:
                severity = "HIGH" if has_previous_defaults else "MODERATE"
                st.warning(f"💰 **DEBT-TO-INCOME: {debt_to_income:.1f}%**")
                st.warning(f"• {severity.lower()} risk - above 30%")
                st.warning("• Requires careful review")
            else:
                st.success(f"💰 **DEBT-TO-INCOME: {debt_to_income:.1f}%**")
                st.success("• Excellent ratio - below 30%")
                st.success("• Strong repayment capacity")
            
            # Credit history assessment
            st.markdown("---")
            if cb_person_cred_hist_length < 2:
                concern = "MAJOR concern" if has_previous_defaults else "warning"
                st.warning(f"📅 **CREDIT HISTORY: {cb_person_cred_hist_length} years**")
                st.warning(f"• Limited history - {concern}")
                st.warning("• Insufficient credit track record")
            else:
                st.success(f"📅 **CREDIT HISTORY: {cb_person_cred_hist_length} years**")
                st.success("• Adequate credit history")
                st.success("• Established credit track record")

    # Real-time metrics
    st.markdown("---")
    st.markdown("### 📊 Real-time Financial Analysis")
    
    if person_income > 0 and loan_amnt > 0:
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                label="Debt-to-Income Ratio", 
                value=f"{debt_to_income:.1f}%",
                delta=f"{'✅ Good' if debt_to_income <= 30 else '⚠️ High'}",
                help="Percentage of income going to debt payments. Lower is better."
            )
            
        with metric_col2:
            monthly_payment = (loan_amnt * (loan_int_rate/100/12)) / (1 - (1 + loan_int_rate/100/12)**(-12*5)) if loan_int_rate > 0 else loan_amnt/60
            monthly_income = person_income / 12
            payment_to_income = (monthly_payment / monthly_income) * 100 if monthly_income > 0 else 0
            
            st.metric(
                label="Est. Monthly Payment", 
                value=f"${monthly_payment:.0f}",
                delta=f"{payment_to_income:.1f}% of monthly income",
                help="Estimated monthly payment for 5-year term"
            )
            
        with metric_col3:
            total_interest = (monthly_payment * 60) - loan_amnt if loan_int_rate > 0 else 0
            interest_rate_category = "Low" if loan_int_rate < 10 else "Medium" if loan_int_rate < 15 else "High"
            
            st.metric(
                label="Total Interest", 
                value=f"${total_interest:.0f}",
                delta=f"{interest_rate_category} rate ({loan_int_rate}%)",
                help="Total interest paid over the life of the loan"
            )
            
        with metric_col4:
            affordability_score = max(0, 100 - debt_to_income - (payment_to_income * 0.5))
            affordability_level = "Excellent" if affordability_score >= 80 else "Good" if affordability_score >= 60 else "Fair" if affordability_score >= 40 else "Poor"
            
            st.metric(
                label="Affordability Score", 
                value=f"{affordability_score:.0f}/100",
                delta=affordability_level,
                help="Overall affordability assessment based on income and debt ratios"
            )
            
        # Detailed breakdown in expandable section
        with st.expander("📈 Detailed Financial Breakdown"):
            breakdown_col1, breakdown_col2 = st.columns(2)
            
            with breakdown_col1:
                st.markdown("**Monthly Budget Impact:**")
                st.write(f"• Monthly Income: ${monthly_income:,.0f}")
                st.write(f"• Loan Payment: ${monthly_payment:.0f}")
                st.write(f"• Remaining Income: ${monthly_income - monthly_payment:,.0f}")
                st.write(f"• Payment Ratio: {payment_to_income:.1f}%")
                
            with breakdown_col2:
                st.markdown("**Loan Cost Analysis:**")
                st.write(f"• Principal Amount: ${loan_amnt:,}")
                st.write(f"• Total Interest: ${total_interest:,.0f}")
                st.write(f"• Total Repayment: ${loan_amnt + total_interest:,.0f}")
                st.write(f"• Interest Rate: {loan_int_rate}%")
                
    else:
        st.info("💡 Enter loan amount and income to see financial analysis")

    # Prediction button
    st.markdown("---")
    predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
    
    with predict_col2:
        if st.button("🔮 Analyze Loan Application", type="primary", use_container_width=True):
            # Preprocessing to match the model's expected format
            try:
                with st.spinner('🤖 AI is analyzing your application...'):
                    time.sleep(1.5)  # Add dramatic effect
                    
                    # Create input DataFrame with preprocessing
                    input_data = pd.DataFrame({
                        'person_age': [person_age],
                        'person_education': [person_education],
                        'person_income': [person_income],
                        'person_emp_exp': [person_emp_exp],
                        'loan_amnt': [loan_amnt],
                        'loan_int_rate': [loan_int_rate],
                        'loan_percent_income': [loan_percent_income],
                        'cb_person_cred_hist_length': [cb_person_cred_hist_length],
                        'credit_score': [credit_score],
                        'previous_loan_defaults_on_file': [previous_loan_defaults_on_file],
                        'person_home_ownership_OWN': [1 if person_home_ownership == 'Own' else 0],
                        'person_home_ownership_RENT': [1 if person_home_ownership == 'Rent' else 0]
                    })
                    
                    # Apply the same preprocessing as in the notebook
                    # 1. Binary encoding for previous_loan_defaults_on_file
                    input_data['previous_loan_defaults_on_file'] = input_data['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})
                    
                    # 2. Ordinal encoding for person_education  
                    education_mapping = {
                        'High School': 1,
                        'Associate': 2,
                        'Bachelor': 3,
                        'Master': 4,
                        'Doctorate': 5
                    }
                    input_data['person_education'] = input_data['person_education'].map(education_mapping)
                    
                    # 3. Ensure we have exactly the columns the model expects (in correct order)
                    # Based on your notebook's final feature set after dropping columns
                    expected_columns = [
                        'person_age', 
                        'person_education', 
                        'person_income', 
                        'person_emp_exp',
                        'loan_amnt', 
                        'loan_int_rate', 
                        'loan_percent_income', 
                        'cb_person_cred_hist_length', 
                        'credit_score', 
                        'previous_loan_defaults_on_file',
                        'person_home_ownership_OWN', 
                        'person_home_ownership_RENT'
                    ]
                    
                    # Reorder columns to match training data
                    input_data_final = input_data[expected_columns]
                    
                    # Verify the final shape matches expectations
                    print(f"Final input shape: {input_data_final.shape}")
                    print(f"Final columns: {list(input_data_final.columns)}")
                    
                    # Make prediction using the pipeline (which handles PCA internally)
                    prediction = pipeline.predict(input_data_final)
                    proba = pipeline.predict_proba(input_data_final)[0][1] * 100
                    
                    # Enhanced results display
                    if prediction[0] == 1:
                        st.markdown(f"""
                        <div class="prediction-approved">
                            <h2>🎉 LOAN APPROVED!</h2>
                            <h3>Approval Probability: {proba:.1f}%</h3>
                            <p>Congratulations! Your loan application has been approved by our AI system.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.balloons()
                        
                    else:
                        st.markdown(f"""
                        <div class="prediction-rejected">
                            <h2>❌ LOAN DECLINED</h2>
                            <h3>Approval Probability: {proba:.1f}%</h3>
                            <p>Unfortunately, your application doesn't meet our current criteria.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Detailed breakdown
                    st.markdown("### 📋 Application Summary")
                    
                    summary_col1, summary_col2 = st.columns(2)
                    
                    with summary_col1:
                        st.markdown("**Personal Details:**")
                        st.write(f"• Age: {person_age} years")
                        st.write(f"• Education: {person_education}")
                        st.write(f"• Income: ${person_income:,}")
                        st.write(f"• Experience: {person_emp_exp} years")
                        
                    with summary_col2:
                        st.markdown("**Loan Details:**")
                        st.write(f"• Amount: ${loan_amnt:,}")
                        st.write(f"• Purpose: {loan_intent}")
                        st.write(f"• Interest Rate: {loan_int_rate}%")
                        st.write(f"• Monthly Payment: ${monthly_payment:.0f}")
                    
                    # Show processed data for debugging
                    with st.expander("🔍 See processed input data (for verification)"):
                        st.markdown("**Data sent to the model:**")
                        st.dataframe(input_data_final, use_container_width=True)
                        st.markdown("**Feature descriptions:**")
                        st.text(f"""
                        • person_age: {person_age}
                        • person_education: {person_education} → {education_mapping[person_education]}
                        • person_income: {person_income}
                        • person_emp_exp: {person_emp_exp}
                        • loan_amnt: {loan_amnt}
                        • loan_int_rate: {loan_int_rate}
                        • loan_percent_income: {loan_percent_income:.4f}
                        • cb_person_cred_hist_length: {cb_person_cred_hist_length}
                        • credit_score: {credit_score}
                        • previous_loan_defaults_on_file: {previous_loan_defaults_on_file} → {input_data_final['previous_loan_defaults_on_file'].iloc[0]}
                        • person_home_ownership_OWN: {input_data_final['person_home_ownership_OWN'].iloc[0]}
                        • person_home_ownership_RENT: {input_data_final['person_home_ownership_RENT'].iloc[0]}
                        """)
                    
                    # Store in session state for history
                    if 'application_history' not in st.session_state:
                        st.session_state.application_history = []
                    
                    st.session_state.application_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'amount': loan_amnt,
                        'purpose': loan_intent,
                        'result': 'Approved' if prediction[0] == 1 else 'Declined',
                        'probability': proba  # Store as percentage (0-100) for display
                    })
                    
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.error("Please check that all fields are filled correctly.")
                
                # Debug information
                with st.expander("🔧 Debug Information"):
                    st.write("Error details:", str(e))
                    if 'input_data_final' in locals():
                        st.write("Final input shape:", input_data_final.shape)
                        st.write("Final input columns:", list(input_data_final.columns))
                    st.write("Expected columns:", expected_columns)

with tab2:
    st.markdown("### 📈 Risk Analysis Dashboard")
    
    if person_income > 0 and loan_amnt > 0:
        # Create risk visualization
        risk_factors = {
            'Credit Score': credit_score / 850 * 100,
            'Income Level': min(100, person_income / 100000 * 100),
            'Employment Experience': min(100, person_emp_exp / 10 * 100),
            'Credit History': min(100, cb_person_cred_hist_length / 20 * 100),
            'Debt-to-Income': max(0, 100 - debt_to_income * 2),
            'No Previous Defaults': 100 if previous_loan_defaults_on_file == 'No' else 0
        }
        
        # Radar chart
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=list(risk_factors.values()),
            theta=list(risk_factors.keys()),
            fill='toself',
            name='Risk Profile',
            line_color='rgb(102, 126, 234)'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            title="Risk Assessment Profile",
            height=500
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Risk score calculation
        overall_risk_score = sum(risk_factors.values()) / len(risk_factors)
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            st.metric("Overall Risk Score", f"{overall_risk_score:.0f}/100")
            
        with risk_col2:
            if overall_risk_score >= 70:
                st.success("🟢 Low Risk Applicant")
            elif overall_risk_score >= 50:
                st.warning("🟡 Medium Risk Applicant")
            else:
                st.error("🔴 High Risk Applicant")
                
        with risk_col3:
            st.metric("Recommended Action", 
                     "Approve" if overall_risk_score >= 60 else "Review/Decline")
        
    else:
        st.info("💡 Fill out the loan application in the first tab to see risk analysis")

with tab3:
    st.markdown("### 📋 Application History")
    
    if 'application_history' in st.session_state and st.session_state.application_history:
        history_df = pd.DataFrame(st.session_state.application_history)
        
        # Summary stats
        total_apps = len(history_df)
        approved_apps = len(history_df[history_df['result'] == 'Approved'])
        avg_amount = history_df['amount'].mean()
        
        hist_col1, hist_col2, hist_col3 = st.columns(3)
        
        with hist_col1:
            st.metric("Total Applications", total_apps)
        with hist_col2:
            st.metric("Approval Rate", f"{(approved_apps/total_apps)*100:.1f}%")
        with hist_col3:
            st.metric("Average Loan Amount", f"${avg_amount:,.0f}")
        
        # History table
        st.dataframe(
            history_df,
            use_container_width=True,
            column_config={
                "timestamp": "Date/Time",
                "amount": st.column_config.NumberColumn("Loan Amount", format="$%d"),
                "purpose": "Purpose",
                "result": "Result",
                "probability": st.column_config.NumberColumn("Probability (%)", format="%.1f%%")
            }
        )
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.application_history = []
            st.rerun()
            
    else:
        st.info("📝 No applications submitted yet. Submit your first application in the Loan Assessment tab!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    🏦 Smart Loan Predictor v2.0 | Powered by Advanced Machine Learning<br>
    <small>This tool provides estimates only. Final decisions subject to full review.</small>
</div>
""", unsafe_allow_html=True)