import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoder
model = joblib.load('../random_forest_house_price_model.pkl')
encoder = joblib.load('../house_price_encoder.pkl')

# Title
st.title("🏠 House Price Prediction App")

# Form for user input
st.subheader("Enter House Features:")
with st.form("prediction_form"):
    MSSubClass = st.selectbox("MSSubClass", [20, 60, 70])
    MSZoning = st.selectbox("MSZoning", ['RL', 'RM', 'C (all)'])
    LotArea = st.number_input("Lot Area (sq ft)", value=8450)
    LotConfig = st.selectbox("Lot Config", ['Inside', 'FR2', 'Corner'])
    BldgType = st.selectbox("Building Type", ['1Fam', 'TwnhsE', 'Duplex'])
    OverallCond = st.slider("Overall Condition (1=Poor, 10=Excellent)", 1, 10, 5)
    YearBuilt = st.number_input("Year Built", value=2000)
    YearRemodAdd = st.number_input("Year Remodeled", value=2000)
    Exterior1st = st.selectbox("Exterior Material", ['VinylSd', 'MetalSd', 'Wd Sdng'])
    BsmtFinSF2 = st.number_input("Basement Finished SF2", value=0)
    TotalBsmtSF = st.number_input("Total Basement SF", value=800)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_dict = {
        'MSSubClass': [MSSubClass],
        'MSZoning': [MSZoning],
        'LotArea': [LotArea],
        'LotConfig': [LotConfig],
        'BldgType': [BldgType],
        'OverallCond': [OverallCond],
        'YearBuilt': [YearBuilt],
        'YearRemodAdd': [YearRemodAdd],
        'Exterior1st': [Exterior1st],
        'BsmtFinSF2': [BsmtFinSF2],
        'TotalBsmtSF': [TotalBsmtSF]
    }

    df_input = pd.DataFrame(input_dict)

    # Encode categorical features
    cat_cols = encoder.feature_names_in_
    encoded = pd.DataFrame(encoder.transform(df_input[cat_cols]), columns=encoder.get_feature_names_out())
    df_input_final = pd.concat([df_input.drop(cat_cols, axis=1), encoded], axis=1)

    # Align columns with training data
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in df_input_final.columns:
            df_input_final[col] = 0  # add missing columns with 0

    df_input_final = df_input_final[model_features]

    prediction = model.predict(df_input_final)[0]
    st.success(f"💰 Estimated Sale Price: ${prediction:,.2f}")
