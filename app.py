
# import streamlit as st
# import pickle
# import os
# import numpy as np
# import pandas as pd
# import time

# # Set page configuration
# st.set_page_config(page_title="Disease Prediction Model", layout="wide", page_icon="")

# def add_logo():
#     # Simple header without logo
#     col1, col2 = st.columns([5, 1.5]) 
#     with col1:
#         st.title("Disease Prediction Model🩺")  

# add_logo()

# # Load the saved models
# try:
#     heart_model = pickle.load(open('Saved_Models/heart_disease_model.sav', 'rb'))
#     heart_scaler = pickle.load(open('Saved_Models/scaler_heart.sav', 'rb'))
# except Exception as e:
#     st.error(f"Error loading heart disease model: {e}")
#     heart_model = None
#     heart_scaler = None

# try:
#     parkinson_model = pickle.load(open('Saved_Models/parkinsons_model.sav', 'rb'))
#     parkinson_scaler = pickle.load(open('Saved_Models/scaler_parkinsons.sav', 'rb'))
# except Exception as e:
#     st.error(f"Error loading Parkinson's model: {e}")
#     parkinson_model = None
#     parkinson_scaler = None

# try:
#     diabetes_model = pickle.load(open('Saved_Models/diabetes_model.sav', 'rb'))
#     diabetes_preprocessing = pickle.load(open('Saved_Models/diabetes_preprocessing.sav', 'rb'))
# except Exception as e:
#     st.error(f"Error loading diabetes model: {e}")
#     diabetes_model = None
#     diabetes_preprocessing = None

# # Function to predict heart disease
# def predict_heart_disease(features):
#     if heart_model is None or heart_scaler is None:
#         return 0  # Default prediction if model isn't loaded
    
#     features_scaled = heart_scaler.transform([features])
#     prediction = heart_model.predict(features_scaled)
#     return prediction[0]  # Return first element of prediction array

# # Function to predict diabetes
# def predict_diabetes(features):
#     """
#     Makes a diabetes prediction based on input features using the full preprocessing pipeline.
    
#     Args:
#         features: List of 8 values in order: [Pregnancies, Glucose, BloodPressure, 
#                  SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    
#     Returns:
#         prediction: 0 (non-diabetic) or 1 (diabetic)
#         probability: Probability of being diabetic (0-1)
#     """
#     if diabetes_model is None or diabetes_preprocessing is None:
#         return 0, 0.0  # Default prediction if model isn't loaded
    
#     try:
#         # Convert list to dictionary with feature names
#         feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
#                          'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
#         patient_data = dict(zip(feature_names, features))
        
#         # Handle zeros in important features where zero is not physiologically possible
#         zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
#         medians = {'Glucose': 117, 'BloodPressure': 72, 'SkinThickness': 23, 
#                   'Insulin': 30, 'BMI': 32}
                  
#         for col in zero_columns:
#             if patient_data[col] == 0:
#                 patient_data[col] = medians[col]
        
#         # Create a DataFrame with the patient data in the correct order
#         df = pd.DataFrame([patient_data])
        
#         # Extract features in the correct order as expected by the model
#         input_data = df[diabetes_preprocessing['original_feature_names']]
        
#         # Apply polynomial features transformation
#         input_poly = diabetes_preprocessing['poly'].transform(input_data)
        
#         # Apply scaling
#         input_scaled = diabetes_preprocessing['scaler'].transform(input_poly)
        
#         # Make prediction
#         prediction = diabetes_model.predict(input_scaled)[0]
#         probability = diabetes_model.predict_proba(input_scaled)[0][1]
        
#         return prediction, probability
    
#     except Exception as e:
#         st.error(f"Diabetes prediction error: {e}")
#         return 0, 0.0  # Default to non-diabetic if there's an error

# # Function to predict Parkinson's disease
# def predict_parkinson(features):
#     if parkinson_model is None or parkinson_scaler is None:
#         return 0  # Default prediction if model isn't loaded
    
#     features_scaled = parkinson_scaler.transform([features])
#     prediction = parkinson_model.predict(features_scaled)
#     return prediction[0]  # Return first element of prediction array

# # App interface
# tabs = st.tabs(["Home", "Heart Disease Prediction", "Diabetes Prediction", "Parkinson's Prediction"])

# with tabs[0]:
#     st.title("Welcome to the Disease Prediction Web App")
#     st.markdown("""
#     ### About the Web App
#     This application uses Machine Learning models to predict the likelihood of:
#     - **Heart Disease**
#     - **Diabetes**
#     - **Parkinson's Disease**
    
#     ### How to Use the Web App
#     1. Navigate to the respective tabs for Heart, Diabetes, or Parkinson's predictions.
#     2. Fill in the required input features in the form.
#     3. Click **Diagnose** to see the result.

#     ### Purpose
#     This app aims to assist medical professionals and individuals in identifying potential risks early, enabling timely medical intervention.
    
#     ### Important Note
#     This application is for educational purposes only and should not replace professional medical advice.
#     """)

# # Heart Disease Prediction Tab
# with tabs[1]:
#     st.header("Heart Disease Prediction🫀")
    
#     if heart_model is None:
#         st.error("Heart disease model could not be loaded. Please check the model files.")
#     else:
#         with st.form(key='heart_form'):
#             # User input fields
#             col1, col2 = st.columns(2)
#             with col1:
#                 age = st.number_input("Age", min_value=0, max_value=100, step=1)
#                 sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
#                 cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
#                                   format_func=lambda x: ["Typical Angina", "Atypical Angina", 
#                                                         "Non-anginal Pain", "Asymptomatic"][x])
#                 trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=0, step=1)
#                 chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, step=1)
#                 fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], 
#                                   format_func=lambda x: "No" if x == 0 else "Yes")
#                 restecg = st.selectbox("Resting ECG Results", [0, 1, 2], 
#                                       format_func=lambda x: ["Normal", "ST-T Wave Abnormality", 
#                                                             "Left Ventricular Hypertrophy"][x])
#             with col2:
#                 thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, step=1)
#                 exang = st.selectbox("Exercise Induced Angina", [0, 1], 
#                                     format_func=lambda x: "No" if x == 0 else "Yes")
#                 oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, step=0.1)
#                 slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], 
#                                     format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x])
#                 ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3])
#                 thal = st.selectbox("Thalassemia", [1, 2, 3], 
#                                    format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1])
            
#             diagnose_button = st.form_submit_button(label="Diagnose")
#             if diagnose_button:
#                 with st.spinner('Analyzing... Please wait.'):
#                     time.sleep(1)  # Simulate processing time
#                     features = [age, sex, cp, trestbps, chol, fbs, restecg, 
#                                thalach, exang, oldpeak, slope, ca, thal]
#                     prediction = predict_heart_disease(features)
#                     if prediction == 1:
#                         st.error("The person has a risk of Heart disease", icon="⚠️")
#                     else:
#                         st.success("The person does not have a risk of Heart disease", icon="✅")

# # Diabetes Prediction Tab
# with tabs[2]:
#     st.header("Diabetes Prediction🩸")
    
#     if diabetes_model is None:
#         st.error("Diabetes model could not be loaded. Please check the model files.")
#     else:
#         with st.form(key='diabetes_form'):
#             # User input fields with better descriptions
#             col1, col2 = st.columns(2)
#             with col1:
#                 pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1)
#                 glucose = st.number_input("Plasma Glucose Concentration (mg/dl)", 
#                                          min_value=0, max_value=300, step=1)
#                 blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", 
#                                                min_value=0, max_value=200, step=1)
#                 skin_thickness = st.number_input("Triceps Skin Fold Thickness (mm)", 
#                                                min_value=0, max_value=100, step=1)
#                 insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", 
#                                          min_value=0, max_value=900, step=1)
#             with col2:
#                 bmi = st.number_input("Body Mass Index (kg/m²)", 
#                                     min_value=0.0, max_value=70.0, step=0.1)
#                 diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 
#                                                   min_value=0.0, max_value=3.0, step=0.01)
#                 age = st.number_input("Age (years)", min_value=0, max_value=120, step=1)

#             diagnose_button = st.form_submit_button(label="Diagnose")
#             if diagnose_button:
#                 with st.spinner('Analyzing... Please wait.'):
#                     time.sleep(1)  # Simulate processing time
#                     features = [pregnancies, glucose, blood_pressure, skin_thickness, 
#                                insulin, bmi, diabetes_pedigree, age]
                    
#                     # Get prediction and probability
#                     prediction, probability = predict_diabetes(features)
                    
#                     # Display risk factors and recommendations
#                     st.subheader("Results")
#                     if prediction == 1:
#                         st.error(f"The person has a risk of Diabetes (Confidence: {probability:.1%})", icon="⚠️")
                        
#                         # Show risk factors
#                         st.subheader("Risk Factors Identified:")
#                         risk_factors = []
                        
#                         if glucose >= 140:
#                             risk_factors.append("High glucose level (≥140 mg/dl). Normal fasting glucose should be below 100 mg/dl.")
#                         elif glucose >= 100:
#                             risk_factors.append("Elevated glucose level (100-139 mg/dl), indicating prediabetes.")
                            
#                         if bmi >= 30:
#                             risk_factors.append("BMI ≥30 indicates obesity, a significant risk factor for diabetes.")
#                         elif bmi >= 25:
#                             risk_factors.append("BMI 25-29.9 indicates overweight, which increases diabetes risk.")
                            
#                         if diabetes_pedigree >= 0.8:
#                             risk_factors.append("High diabetes pedigree function, indicating strong family history.")
                            
#                         if blood_pressure >= 90:
#                             risk_factors.append("Elevated blood pressure (≥90 mm Hg), which often co-occurs with diabetes.")
                            
#                         if not risk_factors:
#                             risk_factors.append("Multiple combined factors contribute to the diabetes risk.")
                            
#                         for factor in risk_factors:
#                             st.warning(factor)
                            
#                     else:
#                         st.success(f"The person does not have a risk of Diabetes (Confidence: {1-probability:.1%})", icon="✅")
                        
#                         # Show preventive advice if some values are borderline
#                         if glucose >= 100 or bmi >= 25:
#                             st.subheader("Preventive Recommendations:")
                            
#                             if glucose >= 100:
#                                 st.info("Your glucose level is borderline. Consider regular monitoring.")
                                
#                             if bmi >= 25:
#                                 st.info("Your BMI indicates overweight. Consider lifestyle modifications to reduce diabetes risk.")
import streamlit as st
import pickle
import os
import numpy as np
import pandas as pd
import time

# Set page configuration
st.set_page_config(
    page_title="Health Guardian - Disease Prediction", 
    layout="wide", 
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# Custom CSS to enhance the UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    .diagnosis-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .result-success {
        background-color: #e7f3ef;
        border-left: 6px solid #2e7d32;
    }
    .result-warning {
        background-color: #fef8e8;
        border-left: 6px solid #f57c00;
    }
    .result-danger {
        background-color: #fdedeb;
        border-left: 6px solid #c62828;
    }
    .risk-factor {
        padding: 12px;
        background-color: #fff3e0;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .recommendation {
        padding: 12px;
        background-color: #e3f2fd;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        border-radius: 6px;
        border: none;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #0D47A1;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .form-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    .feature-section {
        margin-top: 20px;
        border-top: 1px solid #e0e0e0;
        padding-top: 15px;
    }
    .app-footer {
        text-align: center;
        padding: 20px;
        font-size: 0.9rem;
        color: #5f6368;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)

# Application header
def render_header():
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown( "<h1 style='font-size: 60px; text-align: center; color: #1f77b4;'>🧬 Health Guardian</h1>", 
    unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size: 1.1rem; color: #5f6368;">
            Advanced ML-powered disease prediction system
        </p>
        """, unsafe_allow_html=True)

render_header()

# Load the saved models with error handling
@st.cache_resource
def load_models():
    models = {}
    
    try:
        models['heart_model'] = pickle.load(open('Saved_Models/heart_disease_model.sav', 'rb'))
        models['heart_scaler'] = pickle.load(open('Saved_Models/scaler_heart.sav', 'rb'))
    except Exception as e:
        st.error(f"Error loading heart disease model: {e}")
        models['heart_model'] = None
        models['heart_scaler'] = None
    
    try:
        models['parkinson_model'] = pickle.load(open('Saved_Models/parkinsons_model.sav', 'rb'))
        models['parkinson_scaler'] = pickle.load(open('Saved_Models/scaler_parkinsons.sav', 'rb'))
    except Exception as e:
        st.error(f"Error loading Parkinson's model: {e}")
        models['parkinson_model'] = None
        models['parkinson_scaler'] = None
    
    try:
        models['diabetes_model'] = pickle.load(open('Saved_Models/diabetes_model.sav', 'rb'))
        models['diabetes_preprocessing'] = pickle.load(open('Saved_Models/diabetes_preprocessing.sav', 'rb'))
    except Exception as e:
        st.error(f"Error loading diabetes model: {e}")
        models['diabetes_model'] = None
        models['diabetes_preprocessing'] = None
    
    return models

models = load_models()

# Function to predict heart disease
def predict_heart_disease(features):
    if models['heart_model'] is None or models['heart_scaler'] is None:
        return 0  # Default prediction if model isn't loaded
    
    features_scaled = models['heart_scaler'].transform([features])
    prediction = models['heart_model'].predict(features_scaled)
    probability = models['heart_model'].predict_proba(features_scaled)[0][1]
    return prediction[0], probability  # Return prediction and probability

# Function to predict diabetes
def predict_diabetes(features):
    """
    Makes a diabetes prediction based on input features using the full preprocessing pipeline.
    
    Args:
        features: List of 8 values in order: [Pregnancies, Glucose, BloodPressure, 
                 SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    
    Returns:
        prediction: 0 (non-diabetic) or 1 (diabetic)
        probability: Probability of being diabetic (0-1)
    """
    if models['diabetes_model'] is None or models['diabetes_preprocessing'] is None:
        return 0, 0.0  # Default prediction if model isn't loaded
    
    try:
        # Convert list to dictionary with feature names
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                         'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        patient_data = dict(zip(feature_names, features))
        
        # Handle zeros in important features where zero is not physiologically possible
        zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        medians = {'Glucose': 117, 'BloodPressure': 72, 'SkinThickness': 23, 
                  'Insulin': 30, 'BMI': 32}
                  
        for col in zero_columns:
            if patient_data[col] == 0:
                patient_data[col] = medians[col]
        
        # Create a DataFrame with the patient data in the correct order
        df = pd.DataFrame([patient_data])
        
        # Extract features in the correct order as expected by the model
        input_data = df[models['diabetes_preprocessing']['original_feature_names']]
        
        # Apply polynomial features transformation
        input_poly = models['diabetes_preprocessing']['poly'].transform(input_data)
        
        # Apply scaling
        input_scaled = models['diabetes_preprocessing']['scaler'].transform(input_poly)
        
        # Make prediction
        prediction = models['diabetes_model'].predict(input_scaled)[0]
        probability = models['diabetes_model'].predict_proba(input_scaled)[0][1]
        
        return prediction, probability
    
    except Exception as e:
        st.error(f"Diabetes prediction error: {e}")
        return 0, 0.0  # Default to non-diabetic if there's an error

# Function to predict Parkinson's disease
def predict_parkinson(features):
    if models['parkinson_model'] is None or models['parkinson_scaler'] is None:
        return 0, 0.0  # Default prediction if model isn't loaded
    
    features_scaled = models['parkinson_scaler'].transform([features])
    prediction = models['parkinson_model'].predict(features_scaled)
    probability = models['parkinson_model'].predict_proba(features_scaled)[0][1]
    return prediction[0], probability  # Return prediction and probability

# App interface
tabs = st.tabs(["🏠 Home", "❤️ Heart Disease", "🩸 Diabetes", "🧠 Parkinson's"])

with tabs[0]:
    st.markdown('<p class="sub-header">Welcome to Health Guardian</p>', unsafe_allow_html=True)
    
    # Three-column layout for disease types
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: #e3f2fd; padding: 20px; border-radius: 10px; height: 280px;">
            <h3 style="color: #0D47A1;">❤️ Heart Disease</h3>
            <p>Early detection of heart disease can be life-saving. Our model analyzes various cardiac parameters to assess heart disease risk.</p>
            <p><strong>Key indicators:</strong> Blood pressure, cholesterol levels, chest pain type, and ECG results.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: #fff8e1; padding: 20px; border-radius: 10px; height: 280px;">
            <h3 style="color: #FF8F00;">🩸 Diabetes</h3>
            <p>Diabetes affects millions worldwide. Our model examines glucose levels, BMI, and other factors to predict diabetes risk.</p>
            <p><strong>Key indicators:</strong> Glucose concentration, BMI, family history, and insulin levels.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: #f3e5f5; padding: 20px; border-radius: 10px; height: 280px;">
            <h3 style="color: #7B1FA2;">🧠 Parkinson's Disease</h3>
            <p>Parkinson's Disease affects motor function. Our model uses voice pattern analysis to detect early signs.</p>
            <p><strong>Key indicators:</strong> Voice frequency variations, jitter, shimmer, and nonlinear measures.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="sub-header" style="margin-top: 30px;">How it Works</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: flex; justify-content: space-between; margin-top: 20px;">
        <div style="text-align: center; width: 30%;">
            <div style="background-color: #f1f3f4; border-radius: 50%; width: 80px; height: 80px; display: flex; align-items: center; justify-content: center; margin: 0 auto; font-size: 30px;">1</div>
            <h4>Enter Your Data</h4>
            <p>Fill in the required health parameters in the disease-specific tab.</p>
        </div>
        <div style="text-align: center; width: 30%;">
            <div style="background-color: #f1f3f4; border-radius: 50%; width: 80px; height: 80px; display: flex; align-items: center; justify-content: center; margin: 0 auto; font-size: 30px;">2</div>
            <h4>AI Analysis</h4>
            <p>Our machine learning models analyze your data against trained patterns.</p>
        </div>
        <div style="text-align: center; width: 30%;">
            <div style="background-color: #f1f3f4; border-radius: 50%; width: 80px; height: 80px; display: flex; align-items: center; justify-content: center; margin: 0 auto; font-size: 30px;">3</div>
            <h4>Get Results</h4>
            <p>Receive an assessment with risk factors and recommendations.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("""
    **Important Note:** This application is for educational and screening purposes only. 
    The predictions should not replace professional medical diagnosis. 
    Always consult with healthcare professionals for proper medical advice.
    """)

# Heart Disease Prediction Tab
with tabs[1]:
    st.markdown('<p class="sub-header">❤️ Heart Disease Risk Assessment</p>', unsafe_allow_html=True)
    
    if models['heart_model'] is None:
        st.error("Heart disease model could not be loaded. Please check the model files.")
    else:
        # Explanatory text
        st.markdown("""
        Heart disease remains one of the leading causes of death globally. Early detection can significantly 
        improve treatment outcomes. This tool analyzes your cardiac health parameters to assess the risk.
        """)
        
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        with st.form(key='heart_form'):
            # User input fields in a more organized layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<p style="font-weight: 600; color: #0D47A1;">Personal Details</p>', unsafe_allow_html=True)
                age = st.number_input("Age", min_value=0, max_value=100, step=1, value=45)
                sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=1)
                
                st.markdown('<p style="font-weight: 600; color: #0D47A1; margin-top: 20px;">Chest Pain</p>', unsafe_allow_html=True)
                cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                                 format_func=lambda x: ["Typical Angina", "Atypical Angina", 
                                                       "Non-anginal Pain", "Asymptomatic"][x],
                                 index=2)
            
            with col2:
                st.markdown('<p style="font-weight: 600; color: #0D47A1;">Blood Tests</p>', unsafe_allow_html=True)
                trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=0, step=1, value=120)
                chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=0, step=1, value=200)
                fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], 
                                 format_func=lambda x: "No" if x == 0 else "Yes", index=0)
                
                st.markdown('<p style="font-weight: 600; color: #0D47A1; margin-top: 20px;">Cardiac Tests</p>', unsafe_allow_html=True)
                restecg = st.selectbox("Resting ECG Results", [0, 1, 2], 
                                     format_func=lambda x: ["Normal", "ST-T Wave Abnormality", 
                                                           "Left Ventricular Hypertrophy"][x], index=0)
                thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, step=1, value=150)
            
            with col3:
                st.markdown('<p style="font-weight: 600; color: #0D47A1;">Exercise Test Results</p>', unsafe_allow_html=True)
                exang = st.selectbox("Exercise Induced Angina", [0, 1], 
                                   format_func=lambda x: "No" if x == 0 else "Yes", index=0)
                oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, step=0.1, value=0.0)
                slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2], 
                                   format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x], index=1)
                
                st.markdown('<p style="font-weight: 600; color: #0D47A1; margin-top: 20px;">Additional Tests</p>', unsafe_allow_html=True)
                ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3], index=0)
                thal = st.selectbox("Thalassemia", [1, 2, 3], 
                                  format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1], index=1)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                diagnose_button = st.form_submit_button(label="Analyze Heart Health")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Results section
        if 'heart_result' not in st.session_state:
            st.session_state.heart_result = None
            st.session_state.heart_features = None
            st.session_state.heart_probability = None
        
        if diagnose_button:
            with st.spinner('Analyzing cardiac parameters...'):
                time.sleep(1)  # Simulate processing time
                features = [age, sex, cp, trestbps, chol, fbs, restecg, 
                           thalach, exang, oldpeak, slope, ca, thal]
                prediction, probability = predict_heart_disease(features)
                
                st.session_state.heart_result = prediction
                st.session_state.heart_features = features
                st.session_state.heart_probability = probability
        
        # Display results if available
        if st.session_state.heart_result is not None:
            if st.session_state.heart_result == 1:
                st.markdown(f"""
                <div class="diagnosis-box result-danger">
                    <h3>❗ Elevated Heart Disease Risk Detected</h3>
                    <p>Our analysis indicates you may have an increased risk of heart disease. 
                    (Confidence: {st.session_state.heart_probability:.1%})</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk factors analysis
                st.markdown("<h4>Risk Factors Analysis:</h4>", unsafe_allow_html=True)
                
                risk_factors = []
                features = st.session_state.heart_features
                
                if features[3] >= 140:  # trestbps
                    risk_factors.append("High blood pressure (≥140 mmHg)")
                
                if features[4] >= 240:  # chol
                    risk_factors.append("High cholesterol level (≥240 mg/dl)")
                
                if features[2] == 0:  # cp
                    risk_factors.append("Presence of typical angina pain")
                
                if features[8] == 1:  # exang
                    risk_factors.append("Exercise-induced angina")
                
                if features[9] >= 2.0:  # oldpeak
                    risk_factors.append("Significant ST depression (≥2.0)")
                
                if features[12] > 1:  # thal
                    risk_factors.append("Abnormal thalassemia test result")
                
                if features[11] >= 1:  # ca
                    risk_factors.append(f"{features[11]} major vessels colored by fluoroscopy")
                
                if not risk_factors:
                    risk_factors.append("Multiple combined factors contribute to heart disease risk")
                
                for factor in risk_factors:
                    st.markdown(f"""
                    <div class="risk-factor">
                        <strong>⚠️ {factor}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                <div style="margin-top: 20px; padding: 15px; background-color: #ffebee; border-radius: 8px;">
                    <p><strong>Important:</strong> Please consult with a cardiologist for a comprehensive evaluation 
                    and proper diagnosis. This is an AI prediction and not a medical diagnosis.</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown(f"""
                <div class="diagnosis-box result-success">
                    <h3>✅ Low Heart Disease Risk</h3>
                    <p>Based on the provided parameters, we estimate a low risk of heart disease. 
                    (Confidence: {1-st.session_state.heart_probability:.1%})</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Preventive recommendations
                features = st.session_state.heart_features
                if features[3] >= 120 or features[4] >= 200:
                    st.markdown("<h4>Preventive Recommendations:</h4>", unsafe_allow_html=True)
                    
                    if features[3] >= 120:
                        st.markdown("""
                        <div class="recommendation">
                            <strong>📊 Blood Pressure Management</strong>
                            <p>Your blood pressure is slightly elevated. Consider regular monitoring and lifestyle modifications.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if features[4] >= 200:
                        st.markdown("""
                        <div class="recommendation">
                            <strong>📊 Cholesterol Management</strong>
                            <p>Your cholesterol level is borderline. Consider dietary changes and regular check-ups.</p>
                        </div>
                        """, unsafe_allow_html=True)

# Diabetes Prediction Tab
with tabs[2]:
    st.markdown('<p class="sub-header">🩸 Diabetes Risk Assessment</p>', unsafe_allow_html=True)
    
    if models['diabetes_model'] is None:
        st.error("Diabetes model could not be loaded. Please check the model files.")
    else:
        # Explanatory text
        st.markdown("""
        Diabetes is a chronic metabolic condition affecting how the body processes blood sugar. 
        Early detection can help manage the condition effectively and prevent complications.
        """)
        
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        with st.form(key='diabetes_form'):
            # User input fields with better organization
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<p style="font-weight: 600; color: #FF8F00;">Personal Information</p>', unsafe_allow_html=True)
                pregnancies = st.number_input("Number of Pregnancies", min_value=0, step=1, value=0)
                age = st.number_input("Age (years)", min_value=0, max_value=120, step=1, value=35)
                diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 
                                                  min_value=0.0, max_value=3.0, step=0.01, value=0.5,
                                                  help="A measure of diabetes hereditary influence")
                
                st.markdown('<p style="font-weight: 600; color: #FF8F00; margin-top: 20px;">Body Measurements</p>', unsafe_allow_html=True)
                bmi = st.number_input("Body Mass Index (kg/m²)", 
                                    min_value=0.0, max_value=70.0, step=0.1, value=25.0)
                skin_thickness = st.number_input("Triceps Skin Fold Thickness (mm)", 
                                               min_value=0, max_value=100, step=1, value=20)
            
            with col2:
                st.markdown('<p style="font-weight: 600; color: #FF8F00;">Blood Tests</p>', unsafe_allow_html=True)
                glucose = st.number_input("Plasma Glucose Concentration (mg/dl)", 
                                         min_value=0, max_value=300, step=1, value=110)
                blood_pressure = st.number_input("Diastolic Blood Pressure (mm Hg)", 
                                               min_value=0, max_value=200, step=1, value=80)
                insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", 
                                         min_value=0, max_value=900, step=1, value=100)
                
                # Add an information box about glucose values
                st.info("""
                **Reference Values:**
                - Normal fasting glucose: <100 mg/dl
                - Prediabetes: 100-125 mg/dl
                - Diabetes: ≥126 mg/dl
                """)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                diagnose_button = st.form_submit_button(label="Analyze Diabetes Risk")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Results section
        if 'diabetes_result' not in st.session_state:
            st.session_state.diabetes_result = None
            st.session_state.diabetes_features = None
            st.session_state.diabetes_probability = None
        
        if diagnose_button:
            with st.spinner('Analyzing metabolic parameters...'):
                time.sleep(1)  # Simulate processing time
                features = [pregnancies, glucose, blood_pressure, skin_thickness, 
                           insulin, bmi, diabetes_pedigree, age]
                
                # Get prediction and probability
                prediction, probability = predict_diabetes(features)
                
                st.session_state.diabetes_result = prediction
                st.session_state.diabetes_features = features
                st.session_state.diabetes_probability = probability
        
        # Display results if available
        if st.session_state.diabetes_result is not None:
            if st.session_state.diabetes_result == 1:
                st.markdown(f"""
                <div class="diagnosis-box result-danger">
                    <h3>❗ Elevated Diabetes Risk Detected</h3>
                    <p>Our analysis indicates you may have an increased risk of diabetes. 
                    (Confidence: {st.session_state.diabetes_probability:.1%})</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk factors analysis
                st.markdown("<h4>Risk Factors Analysis:</h4>", unsafe_allow_html=True)
                
                risk_factors = []
                features = st.session_state.diabetes_features
                
                if features[1] >= 140:  # glucose
                    risk_factors.append("High glucose level (≥140 mg/dl). Normal fasting glucose should be below 100 mg/dl.")
                elif features[1] >= 100:
                    risk_factors.append("Elevated glucose level (100-139 mg/dl), indicating prediabetes.")
                    
                if features[5] >= 30:  # bmi
                    risk_factors.append("BMI ≥30 indicates obesity, a significant risk factor for diabetes.")
                elif features[5] >= 25:
                    risk_factors.append("BMI 25-29.9 indicates overweight, which increases diabetes risk.")
                    
                if features[6] >= 0.8:  # diabetes_pedigree
                    risk_factors.append("High diabetes pedigree function, indicating strong family history.")
                    
                if features[2] >= 90:  # blood_pressure
                    risk_factors.append("Elevated blood pressure (≥90 mm Hg), which often co-occurs with diabetes.")
                
                if features[7] >= 45:  # age
                    risk_factors.append("Age ≥45 years, which increases diabetes risk.")
                    
                if not risk_factors:
                    risk_factors.append("Multiple combined factors contribute to diabetes risk.")
                
                for factor in risk_factors:
                    st.markdown(f"""
                    <div class="risk-factor">
                        <strong>⚠️ {factor}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("""
                <div style="margin-top: 20px; padding: 15px; background-color: #ffebee; border-radius: 8px;">
                    <p><strong>Important:</strong> Please consult with an endocrinologist for a proper diagnosis 
                    and treatment plan. This is an AI prediction and not a medical diagnosis.</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown(f"""
                <div class="diagnosis-box result-success">
                    <h3>✅ Low Diabetes Risk</h3>
                    <p>Based on the provided parameters, we estimate a low risk of diabetes. 
                    (Confidence: {1-st.session_state.diabetes_probability:.1%})</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Preventive recommendations
                features = st.session_state.diabetes_features
                if features[1] >= 100 or features[5] >= 25:
                    st.markdown("<h4>Preventive Recommendations:</h4>", unsafe_allow_html=True)
                    
                    if features[1] >= 100:
                        st.markdown("""
                        <div class="recommendation">
                            <strong>📊 Glucose Management</strong>
                            <p>Your glucose level is borderline. Consider regular monitoring and dietary modifications.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if features[5] >= 25:
                        st.markdown("""
                        <div class="recommendation">
                            <strong>⚖️ Weight Management</strong>
                            <p>Your BMI indicates overweight. Consider lifestyle modifications to reduce diabetes risk.</p>
                        </div>
                        """, unsafe_allow_html=True)
# Parkinson's Disease Prediction Tab
with tabs[3]:
    st.markdown('<p class="sub-header">🧠 Parkinson\'s Disease Assessment</p>', unsafe_allow_html=True)
    
    if models['parkinson_model'] is None:
        st.error("Parkinson's disease model could not be loaded. Please check the model files.")
    else:
        with st.form(key='parkinson_form'):
            st.write("This tab requires voice measurements taken by specialized equipment.")
            st.write("Enter the voice feature measurements below:")
            
            # Grouped inputs for better organization
            st.subheader("Frequency Measurements")
            col1, col2 = st.columns(2)
            with col1:
                MDVP_Fo_Hz = st.number_input("Average Vocal Fundamental Frequency (Hz)", 
                                           min_value=0.0, step=0.1)
                MDVP_Fhi_Hz = st.number_input("Maximum Vocal Fundamental Frequency (Hz)", 
                                            min_value=0.0, step=0.1)
                MDVP_Flo_Hz = st.number_input("Minimum Vocal Fundamental Frequency (Hz)", 
                                            min_value=0.0, step=0.1)
            
            st.subheader("Jitter Measurements")
            col1, col2 = st.columns(2)  
            with col1:
                MDVP_Jitter = st.number_input("MDVP Jitter (%)", 
                                             min_value=0.0, step=0.001, format="%.6f")
                MDVP_Jitter_Abs = st.number_input("MDVP Absolute Jitter", 
                                                min_value=0.0, step=0.00001, format="%.6f")
                MDVP_RAP = st.number_input("MDVP Relative Amplitude Perturbation", 
                                         min_value=0.0, step=0.001, format="%.6f")
            with col2:
                MDVP_PPQ = st.number_input("MDVP Pitch Period Perturbation Quotient", 
                                         min_value=0.0, step=0.001, format="%.6f")
                Jitter_DDP = st.number_input("Jitter Difference of Differences of Periods", 
                                           min_value=0.0, step=0.001, format="%.6f")
            
            st.subheader("Shimmer Measurements")
            col1, col2 = st.columns(2)
            with col1:
                MDVP_Shim = st.number_input("MDVP Shimmer", 
                                          min_value=0.0, step=0.001, format="%.6f")
                MDVP_Shim_dB = st.number_input("MDVP Shimmer (dB)", 
                                             min_value=0.0, step=0.1)
                Shimmer_APQ3 = st.number_input("Shimmer APQ3", 
                                             min_value=0.0, step=0.001, format="%.6f")
            with col2:
                Shimmer_APQ5 = st.number_input("Shimmer APQ5", 
                                             min_value=0.0, step=0.001, format="%.6f")
                MDVP_APQ = st.number_input("MDVP APQ", 
                                         min_value=0.0, step=0.001, format="%.6f")
                Shimmer_DDA = st.number_input("Shimmer DDA", 
                                            min_value=0.0, step=0.001, format="%.6f")
            
            st.subheader("Noise and Nonlinear Measurements")
            col1, col2 = st.columns(2)
            with col1:
                NHR = st.number_input("Noise-to-Harmonics Ratio", 
                                    min_value=0.0, step=0.001, format="%.6f")
                HNR = st.number_input("Harmonics-to-Noise Ratio", 
                                    min_value=0.0, step=0.1)
                RPDE = st.number_input("Recurrence Period Density Entropy", 
                                     min_value=0.0, max_value=1.0, step=0.001, format="%.6f")
            with col2:
                DFA = st.number_input("Detrended Fluctuation Analysis", 
                                    min_value=0.0, max_value=1.0, step=0.001, format="%.6f")
                spread1 = st.number_input("Spread1", 
                                        min_value=-10.0, max_value=1.0, step=0.001, format="%.6f")
                spread2 = st.number_input("Spread2", 
                                        min_value=-1.0, max_value=1.0, step=0.001, format="%.6f")
                D2 = st.number_input("Correlation Dimension", 
                                   min_value=0.0, step=0.001, format="%.6f")
                PPE = st.number_input("Pitch Period Entropy", 
                                    min_value=0.0, step=0.001, format="%.6f")

            diagnose_button = st.form_submit_button(label="Diagnose")
            if diagnose_button:
                with st.spinner('Analyzing... Please wait.'):
                    time.sleep(1)  # Simulate processing time
                    features = [
                        MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter, MDVP_Jitter_Abs, 
                        MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shim, MDVP_Shim_dB, 
                        Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR, 
                        RPDE, DFA, spread1, spread2, D2, PPE
                    ]
                    prediction = predict_parkinson(features)
                    if prediction == 1:
                        st.error("The person has indicators of Parkinson's Disease", icon="⚠️")
                        st.warning("Please consult with a neurologist for proper diagnosis.")
                    else:
                        st.success("The person does not show indicators of Parkinson's Disease", icon="✅")

# Add footer with disclaimer
st.markdown("---")
st.caption("**Disclaimer**: This application is for educational and research purposes only. " 
          "It should not be used as a substitute for professional medical advice, diagnosis, or treatment.")