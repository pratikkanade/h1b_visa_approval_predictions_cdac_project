import numpy as np
import pickle
import pandas as pd
import streamlit as st
import base64


pickle_in = open("h1b_prediction_model_rf2.pk", "rb")
classifier = pickle.load(pickle_in)


def predict_note_authentication(company_name, soc_name, job_title, salary, city, state):

    prediction = classifier.predict([[company_name, soc_name, job_title, salary, city, state]])
    print(prediction)
    return prediction


def predict_probability(company_name, soc_name, job_title, salary, city, state):
    probability = classifier.predict_proba(np.array([[company_name, soc_name, job_title, salary, city, state]]))
    return probability


df = pd.read_csv("step1_tab.csv")
del df['Unnamed: 0']

df['EMPLOYER_NAME'] = df['EMPLOYER_NAME'].astype("category")
df['SOC_NAME'] = df['SOC_NAME'].astype("category")
df['JOB_TITLE'] = df['JOB_TITLE'].astype("category")
df['STATE'] = df['STATE'].astype("category")
df['CITY'] = df['CITY'].astype("category")

employer_dict = dict(enumerate(df['EMPLOYER_NAME'].cat.categories))
soc_dict = dict(enumerate(df['SOC_NAME'].cat.categories))
jobtitle_dict = dict(enumerate(df['JOB_TITLE'].cat.categories))
state_dict = dict(enumerate(df['STATE'].cat.categories))
city_dict = dict(enumerate(df['CITY'].cat.categories))


st.title("H1B Visa Prediction")
#    html_temp = """
#    <div style="background-color:black;padding:5px">
#    <h2 style="color:white;text-align:center;"> H1B Visa Approval Prediction </h2>
#    </div>
#    """


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
            background-attachment: fixed;
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


add_bg_from_local('background2.jpg')

col1, col2, = st.columns(2)

with col1:
    company_name = int(list(employer_dict.values()).index(st.selectbox("Company Name", employer_dict.values())))
    soc_name = int(list(soc_dict.values()).index(st.selectbox("SOC Name", soc_dict.values())))
    job_title = int(list(jobtitle_dict.values()).index(st.selectbox("Job Title", jobtitle_dict.values())))


with col2:
    state_text = st.selectbox("State", state_dict.values())
    state = int(list(state_dict.values()).index(state_text))

    city_options = {

        'ALABAMA': list(df.loc[df['STATE'] == 'ALABAMA', 'CITY'].drop_duplicates()),
        'MICHIGAN': list(df.loc[df['STATE'] == 'MICHIGAN', 'CITY'].drop_duplicates()),
        'ARKANSAS': list(df.loc[df['STATE'] == 'ARKANSAS', 'CITY'].drop_duplicates()),
        'CALIFORNIA': list(df.loc[df['STATE'] == 'CALIFORNIA', 'CITY'].drop_duplicates()),
        'MINNESOTA': list(df.loc[df['STATE'] == 'MINNESOTA', 'CITY'].drop_duplicates()),
        'MASSACHUSETTS': list(df.loc[df['STATE'] == 'MASSACHUSETTS', 'CITY'].drop_duplicates()),
        'NEW YORK': list(df.loc[df['STATE'] == 'NEW YORK', 'CITY'].drop_duplicates()),
        'PENNSYLVANIA': list(df.loc[df['STATE'] == 'PENNSYLVANIA', 'CITY'].drop_duplicates()),
        'INDIANA': list(df.loc[df['STATE'] == 'INDIANA', 'CITY'].drop_duplicates()),
        'WASHINGTON': list(df.loc[df['STATE'] == 'WASHINGTON', 'CITY'].drop_duplicates()),
        'NEW JERSEY': list(df.loc[df['STATE'] == 'NEW JERSEY', 'CITY'].drop_duplicates()),
        'TEXAS': list(df.loc[df['STATE'] == 'TEXAS', 'CITY'].drop_duplicates()),
        'MARYLAND': list(df.loc[df['STATE'] == 'MARYLAND', 'CITY'].drop_duplicates()),
        'ILLINOIS': list(df.loc[df['STATE'] == 'ILLINOIS', 'CITY'].drop_duplicates()),
        'NORTH CAROLINA': list(df.loc[df['STATE'] == 'NORTH CAROLINA', 'CITY'].drop_duplicates()),
        'OHIO': list(df.loc[df['STATE'] == 'OHIO', 'CITY'].drop_duplicates()),
        'FLORIDA': list(df.loc[df['STATE'] == 'FLORIDA', 'CITY'].drop_duplicates()),
        'GEORGIA': list(df.loc[df['STATE'] == 'GEORGIA', 'CITY'].drop_duplicates()),
        'COLORADO': list(df.loc[df['STATE'] == 'COLORADO', 'CITY'].drop_duplicates()),
        'KENTUCKY': list(df.loc[df['STATE'] == 'KENTUCKY', 'CITY'].drop_duplicates()),
        'OREGON': list(df.loc[df['STATE'] == 'OREGON', 'CITY'].drop_duplicates()),
        'MISSOURI': list(df.loc[df['STATE'] == 'MISSOURI', 'CITY'].drop_duplicates()),
        'VIRGINIA': list(df.loc[df['STATE'] == 'VIRGINIA', 'CITY'].drop_duplicates()),
        'WISCONSIN': list(df.loc[df['STATE'] == 'WISCONSIN', 'CITY'].drop_duplicates()),
        'ARIZONA': list(df.loc[df['STATE'] == 'ARIZONA', 'CITY'].drop_duplicates()),
        'CONNECTICUT': list(df.loc[df['STATE'] == 'CONNECTICUT', 'CITY'].drop_duplicates()),
        'TENNESSEE': list(df.loc[df['STATE'] == 'TENNESSEE', 'CITY'].drop_duplicates()),
        'NEW HAMPSHIRE': list(df.loc[df['STATE'] == 'NEW HAMPSHIRE', 'CITY'].drop_duplicates()),
        'RHODE ISLAND': list(df.loc[df['STATE'] == 'RHODE ISLAND', 'CITY'].drop_duplicates()),
        'UTAH': list(df.loc[df['STATE'] == 'UTAH', 'CITY'].drop_duplicates()),
        'DISTRICT OF COLUMBIA': list(df.loc[df['STATE'] == 'DISTRICT OF COLUMBIA', 'CITY'].drop_duplicates()),
        'NEVADA': list(df.loc[df['STATE'] == 'NEVADA', 'CITY'].drop_duplicates()),
        'NEBRASKA': list(df.loc[df['STATE'] == 'NEBRASKA', 'CITY'].drop_duplicates()),
        'KANSAS': list(df.loc[df['STATE'] == 'KANSAS', 'CITY'].drop_duplicates()),
        'DELAWARE': list(df.loc[df['STATE'] == 'DELAWARE', 'CITY'].drop_duplicates()),
        'IOWA': list(df.loc[df['STATE'] == 'IOWA', 'CITY'].drop_duplicates()),
        'SOUTH CAROLINA': list(df.loc[df['STATE'] == 'SOUTH CAROLINA', 'CITY'].drop_duplicates()),
        'IDAHO': list(df.loc[df['STATE'] == 'IDAHO', 'CITY'].drop_duplicates()),
        'LOUISIANA': list(df.loc[df['STATE'] == 'LOUISIANA', 'CITY'].drop_duplicates()),
        'OKLAHOMA': list(df.loc[df['STATE'] == 'OKLAHOMA', 'CITY'].drop_duplicates()),
        'MAINE': list(df.loc[df['STATE'] == 'MAINE', 'CITY'].drop_duplicates()),
        'VERMONT': list(df.loc[df['STATE'] == 'VERMONT', 'CITY'].drop_duplicates())

    }

    city = int(list(city_dict.values()).index(st.selectbox("City", city_options[state_text])))
    salary = st.number_input("Salary")

result = ""
if st.button("Predict"):
    approval = predict_note_authentication(company_name, soc_name, job_title, salary, city, state)
    probability = predict_probability(company_name, soc_name, job_title, salary, city, state)

    if approval[0] >= 0.5:
        result = f"There is a {round(probability[0][1] * 100, 2)}% probability that your visa might be approved for the random selection process."
    else:
        result = f"There is a {round(probability[0][0] * 100, 2)}% probability that your visa might not be approved for the random selection process."


#        if approval[0] == 0:
#            result = 'Your visa might not be approved.'
#            probability = round(predict_probability[0][0] * 100, 3)
#        else:
#            result = 'Your visa might be approved.'
#            probability = round(predict_probability[0][1] * 100, 3)

st.success(format(result))

