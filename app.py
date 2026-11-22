import os

import requests
import streamlit as st

# address of the FastAPI app (local uvicorn or the Docker container)
api_url = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="German Credit Risk")
st.title("German Credit Risk")
st.write("Fill in the loan application. The form sends it to the FastAPI `/predict` endpoint "
         "and shows if the client is a good or a bad risk.")

# the frontend does not load the model, it only talks to the API
try:
    info = requests.get(f"{api_url}/", timeout=5).json()
    schema = requests.get(f"{api_url}/openapi.json", timeout=5).json()
except requests.exceptions.ConnectionError:
    st.error(f"The API is not running at {api_url}. Start it first (uvicorn or Docker).")
    st.stop()

st.caption(f"API: {api_url} | model: {info['model_name']}, version {info['model_version']} (MLflow Model Registry)")

# allowed values of every feature are taken from the API schema, so they are the same as in main.py
fields = schema["components"]["schemas"]["CreditApplication"]["properties"]
example = schema["components"]["schemas"]["CreditApplication"]["examples"][0]

sections = {
    "Loan": ["duration", "credit_amount", "purpose", "installment_rate",
             "credit_history", "existing_credits", "other_installment_plans", "other_debtors"],
    "Client": ["age", "personal_status", "job", "employment", "people_liable", "telephone", "foreign_worker"],
    "Money and housing": ["checking_status", "savings", "property", "housing", "residence_since"]
}

application = {}
with st.form("application"):
    for section, names in sections.items():
        st.subheader(section)
        cols = st.columns(2)
        for i, name in enumerate(names):
            field = fields[name]
            label = name.replace("_", " ").capitalize()
            with cols[i % 2]:
                if "enum" in field:
                    # categorical feature -> dropdown, the first client from the dataset is the default
                    options = field["enum"]
                    application[name] = st.selectbox(label, options, index=options.index(example[name]))
                elif field["type"] == "integer":
                    # the limits come from the schema as floats (for example 4.0), so they are changed to int
                    max_value = int(field["maximum"]) if "maximum" in field else None
                    application[name] = st.number_input(label, min_value=int(field.get("minimum", 1)),
                                                        max_value=max_value, value=example[name],
                                                        step=1, help=field.get("description"))
                else:
                    application[name] = st.number_input(label, min_value=1.0, value=float(example[name]),
                                                        step=100.0, help=field.get("description"))

    submitted = st.form_submit_button("Predict")

if submitted:
    response = requests.post(f"{api_url}/predict", json=application, timeout=10)

    if response.status_code != 200:
        st.error(f"API error {response.status_code}: {response.json()}")
    else:
        result = response.json()
        prob = result["probability_bad"]

        if result["risk"] == "bad":
            st.error(f"Bad risk: the model would reject this application (probability of bad risk {prob:.1%})")
        else:
            st.success(f"Good risk: the model would approve this application (probability of bad risk {prob:.1%})")

        st.progress(prob, text=f"Probability of bad risk: {prob:.1%} (bad risk if it is above 50%)")

        with st.expander("Response from the API"):
            st.json(result)
