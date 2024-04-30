import streamlit as st
import pandas as pd
import joblib

# Load the trained model and feature names
model = joblib.load('insurance_claim_predictor_model.pkl')
feature_names = joblib.load('feature_names.pkl')

def preprocess_input(user_input, feature_names):
    # Create a DataFrame for the user input
    input_df = pd.DataFrame([user_input])

    # Process categorical features: Convert to dummies
    input_df = pd.get_dummies(input_df)

    # Add missing dummy columns that the model is expecting
    missing_cols = set(feature_names) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    # Ensure the columns are ordered correctly
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    return input_df

def user_input_features():
    st.write("Please input data for prediction:")
    user_input = {}

    # Collect input for each feature
    user_input['Carrier'] = st.selectbox('Carrier', ['Carrier A', 'Carrier B', 'Carrier C'])
    user_input['Year'] = st.number_input('Year', min_value=2000, max_value=2024, value=2022)
    user_input['Service category'] = st.text_input('Service Category')
    user_input['Request'] = st.text_area('Request Description')
    user_input['Code type'] = st.text_input('Code Type')
    user_input['Code'] = st.text_input('Code')
    user_input['Description of service'] = st.text_area('Description of Service')
    user_input['Number of requests per code'] = st.number_input('Number of Requests per Code', min_value=0)
    user_input['Initially denied then approved - approval rate'] = st.slider('Initially Denied Then Approved - Approval Rate', 0.0, 1.0, 0.5)
    user_input['Expedited - Avg response time'] = st.number_input('Expedited - Avg Response Time')
    user_input['Standard - Avg response time'] = st.number_input('Standard - Avg Response Time')
    user_input['Extenuating circumstances - Avg response time'] = st.number_input('Extenuating Circumstances - Avg Response Time')
    user_input['Expedited - Number of requests'] = st.number_input('Expedited - Number of Requests')
    user_input['Standard - Number of requests'] = st.number_input('Standard - Number of Requests')
    user_input['Extenuating circumstances - Number of requests'] = st.number_input('Extenuating Circumstances - Number of Requests')
    user_input['Drug class'] = st.text_input('Drug Class')
    user_input['Drug name'] = st.text_input('Drug Name')
    user_input['Drug code'] = st.text_input('Drug Code')

    # Convert user input to DataFrame in the same format as the model's training data
    input_data = preprocess_input(user_input, feature_names)
    return input_data

def main():
    st.title('Insurance Claim Predictor')
    
    # Display the input fields in a form
    with st.form(key='prediction_form'):
        input_df = user_input_features()
        submit_button = st.form_submit_button(label='Predict')
    
    if submit_button:
        # Make prediction and display results
        prediction_proba = model.predict_proba(input_df)
        
        # Adjust the decision threshold
        decision_threshold = st.slider('Decision Threshold', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        st.write(f"Probability of being Approved: {prediction_proba[0][1]:.4f}")
        st.write(f"Probability of being Not Approved: {prediction_proba[0][0]:.4f}")
        
        # Display the prediction outcome based on the threshold
        result = 'Approved' if prediction_proba[0][1] > decision_threshold else 'Not Approved'
        st.subheader('Predicted Outcome:')
        st.success(result)

if __name__ == '__main__':
    main()

