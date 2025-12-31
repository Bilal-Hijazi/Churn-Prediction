import streamlit as st
import joblib
import pandas as pd
from pipeline import data

st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

st.title("Telco Churn Prediction")
st.caption("Enter the information related to your customer to predict churn")
def load_model():

    return joblib.load("churn_model.pkl")


def get_unique_values(data: pd.DataFrame, column:str):
    return sorted(data[column].dropna().unique().tolist())

left, right = st.columns([1.3, 1])
with left:

    simple_data_needed_from_user = {
        'Tenure (Months)': [None],
        'Contract': [None],
        'Internet Service': [None],
        'Online Security': [None]
    }
    column_configs={
        "Contract": st.column_config.SelectboxColumn("Contract",options=get_unique_values(data,"Contract")),
        "Tenure (Months)": st.column_config.NumberColumn(),
        "Internet Service": st.column_config.SelectboxColumn("Internet Service", options=get_unique_values(data,"InternetService")),
        "Online Security": st.column_config.SelectboxColumn("Online Security",options=get_unique_values(data, "OnlineSecurity"))
    }
    important_features_entred_by_user = pd.DataFrame(simple_data_needed_from_user)
    user_input = st.data_editor(important_features_entred_by_user,column_config=column_configs)

    table_to_give_model = pd.DataFrame(columns=data.columns)
    table_to_give_model.loc[0] = None
    table_to_give_model = table_to_give_model.drop('Churn',axis=1)
    table_to_give_model.loc[0, "Tenure (Months)"] = user_input.loc[0, "Tenure (Months)"]
    table_to_give_model.loc[0, "Contract"] = user_input.loc[0, "Contract"]
    table_to_give_model.loc[0, "InternetService"] = user_input.loc[0, "Internet Service"]
    table_to_give_model.loc[0, "OnlineSecurity"] = user_input.loc[0, "Online Security"]

    predict_button = st.button("Show Churn Risk")

with right:

    st.subheader("Results")
    proba = 0

    if predict_button:
        model = load_model()
        proba = model.predict_proba(table_to_give_model)[0,1]
        st.markdown(f"### Churn probability: **{proba:.1%}**")
    if proba == 0:
        risk = 'None'
        rec = 'None'
    elif proba > 0 and proba <0.30:
        risk = "Low"
        rec = "No action needed"
    elif proba < 0.60:
        risk = "Medium"
        rec = "Consider monitoring OR light retention offer"
    else:
        risk = "High"
        rec = "Recommend contacting with retention offer"

    st.write(f"**Risk level:** {risk}")
    st.write(f"**Recommendation:** {rec}")

with st.expander("About this project:"):
    st.write("Random Forest model trained on Telco churn dataset. Outputs churn probability (not certainty).")
    st.write("The features in the table are the most important features to the RFC to make a decision. The rest of the features it needs to output probability are filled automatically.")
    st.write('Any fields left empty are filled by the model')