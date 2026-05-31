
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title='Tourism Wellness Package Prediction', layout='centered')
st.title('Wellness Tourism Package Purchase Prediction')
st.write('This app predicts whether a customer is likely to purchase the Wellness Tourism Package.')

# Load trained model
@st.cache_resource
def load_model():
    model_path = 'best_model.joblib'
    model = joblib.load(model_path)
    return model

model = load_model()

# Collect user inputs
def user_input_features():
    age = st.number_input('Age', min_value=18, max_value=80, value=35)
    typeofcontact = st.selectbox('Type of Contact', ['Company Invited', 'Self Enquiry'])
    citytier = st.selectbox('City Tier', [1, 2, 3])
    durationofpitch = st.number_input('Duration Of Pitch (minutes)', min_value=0.0, max_value=60.0, value=10.0)
    occupation = st.selectbox('Occupation', ['Salaried', 'Small Business', 'Free Lancer', 'Large Business'])
    gender = st.selectbox('Gender', ['Male', 'Female'])
    numberofpersonvisiting = st.number_input('Number Of Person Visiting', min_value=1, max_value=10, value=2)
    numberoffollowups = st.number_input('Number Of Followups', min_value=0, max_value=10, value=2)
    productpitched = st.selectbox('Product Pitched', ['Basic', 'Standard', 'Deluxe', 'Super Deluxe', 'King'])
    preferredpropertystar = st.selectbox('Preferred Property Star', [1, 2, 3, 4, 5])
    maritalstatus = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Unmarried'])
    numberoftrips = st.number_input('Number Of Trips per year', min_value=0.0, max_value=20.0, value=2.0)
    passport = st.selectbox('Passport', [0, 1])
    pitchsatisfactionscore = st.selectbox('Pitch Satisfaction Score', [1, 2, 3, 4, 5])
    owncar = st.selectbox('Own Car', [0, 1])
    numberofchildrenvisiting = st.number_input('Number Of Children Visiting', min_value=0.0, max_value=10.0, value=0.0)
    designation = st.selectbox('Designation', ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])
    monthlyincome = st.number_input('Monthly Income', min_value=5000.0, max_value=100000.0, value=20000.0)

    data = {
        'Age': age,
        'TypeofContact': typeofcontact,
        'CityTier': citytier,
        'DurationOfPitch': durationofpitch,
        'Occupation': occupation,
        'Gender': gender,
        'NumberOfPersonVisiting': numberofpersonvisiting,
        'NumberOfFollowups': numberoffollowups,
        'ProductPitched': productpitched,
        'PreferredPropertyStar': preferredpropertystar,
        'MaritalStatus': maritalstatus,
        'NumberOfTrips': numberoftrips,
        'Passport': passport,
        'PitchSatisfactionScore': pitchsatisfactionscore,
        'OwnCar': owncar,
        'NumberOfChildrenVisiting': numberofchildrenvisiting,
        'Designation': designation,
        'MonthlyIncome': monthlyincome
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

if st.button('Predict'):
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]
    if prediction == 1:
        st.success(f'Customer is LIKELY to purchase the Wellness Package (probability {prediction_proba:.2f}).')
    else:
        st.warning(f'Customer is UNLIKELY to purchase the Wellness Package (probability {prediction_proba:.2f}).')

    st.write('Input summary:')
    st.dataframe(input_df)
