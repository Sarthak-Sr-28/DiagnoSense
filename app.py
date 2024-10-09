import os
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
import regex as re

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

    
# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open("E:\Multiple Disease Prediction\Saved Models\diabetes_model.sav", 'rb'))

heart_disease_model = pickle.load(open("E:\Multiple Disease Prediction\Saved Models\heart_disease_model.sav", 'rb'))

parkinsons_model = pickle.load(open("E:\Multiple Disease Prediction\Saved Models\parkinson_disease_model.sav", 'rb'))

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',

                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person'],
                           default_index=0)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.header('Diabetes Prediction using ML')
    # Input fields for diabetes prediction
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.number_input('Number of Pregnancies', min_value=0, format='%d')
    with col2:
        Glucose = st.number_input('Glucose Level', min_value=0, format='%d')
    with col3:
        BloodPressure = st.number_input('Blood Pressure value', min_value=0, format='%d')
    with col1:
        SkinThickness = st.number_input('Skin Thickness value', min_value=0, format='%d')
    with col2:
        Insulin = st.number_input('Insulin Level', min_value=0, format='%d')
    with col3:
        BMI = st.number_input('BMI value', min_value=0.0, format='%.2f')
    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.000, format='%.3f')
    with col2:
        Age = st.number_input('Age of the Person', min_value=0, format='%d')

    diab_diagnosis = ''

    # Prediction button for diabetes
    if st.button('Diabetes Test Result'):
        try:
            # Validate inputs
            if None in [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]:
                st.error("Please fill out all fields before submitting.")
            else:
                # Feature engineering
                NewBMI_Underweight = 1 if BMI < 18.5 else 0
                NewBMI_Overweight = 1 if 24.9 < BMI <= 29.9 else 0
                NewBMI_Obesity_1 = 1 if 29.9 < BMI <= 34.9 else 0
                NewBMI_Obesity_2 = 1 if 34.9 < BMI <= 39.9 else 0
                NewBMI_Obesity_3 = 1 if BMI > 39.9 else 0
                NewInsulinScore_Normal = 1 if 16 <= Insulin <= 166 else 0
                NewGlucose_Low = 1 if Glucose <= 70 else 0
                NewGlucose_Normal = 1 if 70 < Glucose <= 99 else 0
                NewGlucose_Overweight = 1 if 99 < Glucose <= 126 else 0

                # Inputs
                user_input_categorised = [
                    Pregnancies, NewBMI_Obesity_1, NewBMI_Obesity_2,
                    NewBMI_Obesity_3, NewBMI_Overweight, NewBMI_Underweight,
                    NewInsulinScore_Normal, NewGlucose_Low, NewGlucose_Normal,
                    NewGlucose_Overweight
                ]

                user_input_continuous = np.array([Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]).reshape(1, -1)

                # Load scaler and quantile transformer
                sc = pickle.load(open("E:/Multiple Disease Prediction/Saved Scalers/diabetes_scaler.pkl", 'rb'))
                qt = pickle.load(open("E:/Multiple Disease Prediction/Saved Quantile Transformer/diabetes_quantile.pkl", 'rb'))

                # 1. Scale the entire data with the StandardScaler
                user_input_standardized = sc.transform(user_input_continuous)  # This expects a shape of (1, 7) - all features

                # 2. Apply quantile transformer feature-wise (column-wise)
                user_input_transformed = np.empty_like(user_input_standardized)  # Create an empty array to store transformed data

                for i in range(user_input_standardized.shape[1]):  # Iterate through each column (feature)
                    user_input_transformed[:, i] = qt.transform(user_input_standardized[:, i].reshape(-1, 1)).flatten()

                # Convert transformed continuous inputs to a list
                user_input_continuous_list = user_input_transformed.flatten().tolist()

                # Merge continuous inputs with categorized inputs
                final_user_input = user_input_categorised + user_input_continuous_list
                # Prediction
                diab_prediction = diabetes_model.predict([final_user_input])
                diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'

        except Exception as e:
            st.error(f"Input Error: {e}")

    st.success(diab_diagnosis)


# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=0, format="%d")

    with col2:
        sex = st.number_input('Sex: 1 = Male; 0 = Female', min_value=0, max_value=1, format="%d")

    with col3:
        cp = st.number_input('Chest Pain types: 0 = typical angina; 1 = atypical angina; 2 = non-anginal pain; 3 = asymptomatic', min_value=0, max_value=3, format="%d")

    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=0, format="%d")

    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', min_value=0, format="%d")

    with col3:
        fbs = st.number_input('Fasting Blood Sugar > 120 mg/dl: 1 = true; 0 = false', min_value=0, max_value=1, format="%d")

    with col1:
        restecg = st.number_input('Resting Electrocardiographic results: 0 = normal; 1 = having ST-T wave abnormality; 2 = showing probable or definite left ventricular hypertrophy by Estes \'criteria\'', min_value=0, max_value=2, format="%d")

    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved',  min_value=0, format="%d")

    with col3:
        exang = st.number_input('Exercise Induced Angina: 1 = yes, 0 = no', min_value=0, max_value=1, format="%d")

    with col1:
        oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, format = "%.1f")

    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment: 0 = upsloping; 1 = flat; 2 = downsloping', min_value=0, max_value=2, format='%d')

    with col3:
        ca = st.number_input('Major vessels colored by flourosopy', min_value=0, max_value=3, format="%d")

    with col1:
        thal = st.number_input('thal: 1 = normal; 2 = fixed defect; 3 = reversable defect', min_value=1, max_value=3, format='%d')
    
    


    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):
        try:
            if None in [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]:
                st.error("Please fill out all fields before submitting.")
            else:
                if chol < 200:
                    NewChol_Normal = 1
                else:
                    NewChol_Normal = 0
                
                if chol > 240:
                    NewChol_High = 1
                else:
                    NewChol_High = 0

                if trestbps < 120:
                    NewTrestbps_Normal = 1
                else:
                    NewTrestbps_Normal = 0
                
                if trestbps >= 130 and trestbps <= 139:
                    NewTrestbps_Stage_1_Hypertension = 1
                else:
                    NewTrestbps_Stage_1_Hypertension = 0

                if trestbps >= 140:
                    NewTrestbps_Stage_2_Hypertension = 1
                else:
                    NewTrestbps_Stage_2_Hypertension = 0

        
                user_input_categorised = [sex, cp, fbs, restecg, exang, slope, ca, thal, NewChol_High,
                       NewChol_Normal, NewTrestbps_Normal, NewTrestbps_Stage_1_Hypertension,
                         NewTrestbps_Stage_2_Hypertension]
                
                user_input_continuous = np.array([age, trestbps, chol, thalach, oldpeak]).reshape(1, -1)

                # Load scaler and quantile transformer
                sc = pickle.load(open("E:/Multiple Disease Prediction/Saved Scalers/heart_scaler.pkl", 'rb'))
                qt = pickle.load(open("E:/Multiple Disease Prediction/Saved Quantile Transformer/heart_quantile.pkl", 'rb'))

                # 1. Scale the entire data with the StandardScaler
                user_input_standardized = sc.transform(user_input_continuous)  

                # 2. Apply quantile transformer feature-wise (column-wise)
                user_input_transformed = np.empty_like(user_input_standardized)  # Create an empty array to store transformed data

                for i in range(user_input_standardized.shape[1]):  # Iterate through each column (feature)
                    user_input_transformed[:, i] = qt.transform(user_input_standardized[:, i].reshape(-1, 1)).flatten()

                # Convert transformed continuous inputs to a list
                user_input_continuous_list = user_input_transformed.flatten().tolist()

                # Merge continuous inputs with categorized inputs
                final_user_input = user_input_categorised + user_input_continuous_list

                heart_prediction = heart_disease_model.predict([final_user_input])

                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The person is having heart disease'
                else:
                    heart_diagnosis = 'The person does not have any heart disease'

        except Exception as e:
            st.error(f"Error: {e}")

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, format="%.3f")

    with col2:
        fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, format="%.3f")

    with col3:
        flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, format="%.3f")

    with col4:
        Jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, format="%.5f")

    with col5:
        Jitter_Abs = st.number_input('MDVP:Jitter(Abs)', min_value = 0.0, format="%.5f")

    with col1:
        RAP = st.number_input('MDVP:RAP', min_value=0.0, format="%.5f" )

    with col2:
        PPQ = st.number_input('MDVP:PPQ', min_value=0.0, format="%.5f")

    with col3:
        DDP = st.number_input('Jitter:DDP',min_value=0.0, format="%.5f")

    with col4:
        Shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, format="%.5f")

    with col5:
        Shimmer_dB = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, format="%.3f")

    with col1:
        APQ3 = st.number_input('Shimmer:APQ3', min_value=0.0, format="%.5f")

    with col2:
        APQ5 = st.number_input('Shimmer:APQ5', min_value=0.0, format="%.5f")

    with col3:
        APQ = st.number_input('MDVP:APQ', min_value=0.0, format="%.5f")

    with col4:
        DDA = st.number_input('Shimmer:DDA', min_value=0.0, format="%.5f")

    with col5:
        NHR = st.number_input('NHR', min_value=0.0, format="%.5f")

    with col1:
        HNR = st.number_input('HNR', min_value=0.0, format="%.3f")

    with col2:
        RPDE = st.number_input('RPDE', min_value=0.0, format="%.6f")

    with col3:
        DFA = st.number_input('DFA', min_value=0.0, format="%.6f")

    with col4:
        spread1 = st.number_input('spread1', format="%.6f")

    with col5:
        spread2 = st.number_input('spread2', format="%.6f")

    with col1:
        D2 = st.number_input('D2', format="%.6f")

    with col2:
        PPE = st.number_input('PPE', format="%.6f")

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        try:
            user_input = np.array([fo, fhi, flo, Jitter_percent, Jitter_Abs,
                        RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                        APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]).reshape(1,-1)
            
            # Load scaler and quantile transformer
            sc = pickle.load(open("E:/Multiple Disease Prediction/Saved Scalers/parkinson_scaler.pkl", 'rb'))
            qt = pickle.load(open("E:/Multiple Disease Prediction/Saved Quantile Transformer/parkinson_quantile.pkl", 'rb'))

            # 1. Scale the entire data with the StandardScaler
            user_input_standardized = sc.transform(user_input)  

            # 2. Apply quantile transformer feature-wise (column-wise)
            user_input_transformed = np.empty_like(user_input_standardized)  # Create an empty array to store transformed data

            for i in range(user_input_standardized.shape[1]):  # Iterate through each column (feature)
                user_input_transformed[:, i] = qt.transform(user_input_standardized[:, i].reshape(-1, 1)).flatten()

            # Convert transformed continuous inputs to a list
            final_user_input = user_input_transformed.flatten().tolist()

            parkinsons_prediction = parkinsons_model.predict([final_user_input])

            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
        except Exception as e:
            st.error(f'Error: {e}')
    st.success(parkinsons_diagnosis)
