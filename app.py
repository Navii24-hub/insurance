
import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Insurance Charges Prediction",
    page_icon="🏥",
    layout="centered"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# Application title
st.title("🏥 Insurance Charges Prediction")

st.write(
    "Enter your personal and insurance details "
    "to predict your estimated medical insurance charges."
)

st.divider()

# User inputs
st.subheader("Enter Your Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25,
        step=1
    )

    sex = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )

with col2:
    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    smoker = st.selectbox(
        "Smoking Status",
        ["No", "Yes"]
    )

    region = st.selectbox(
        "Region",
        ["Northeast", "Northwest", "Southeast", "Southwest"]
    )

st.divider()

# Prediction button
if st.button(
    "Predict Insurance Charges",
    type="primary",
    use_container_width=True
):

    # Encoding categorical values
    sex_encoded = 0 if sex == "Female" else 1

    smoker_encoded = 0 if smoker == "No" else 1

    region_mapping = {
        "Northeast": 0,
        "Northwest": 1,
        "Southeast": 2,
        "Southwest": 3
    }

    region_encoded = region_mapping[region]

    # Create input DataFrame
    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex_encoded,
        "bmi": bmi,
        "children": children,
        "smoker": smoker_encoded,
        "region": region_encoded
    }])

    try:
        prediction = model.predict(input_data)[0]

        st.success("Prediction Completed Successfully!")

        st.subheader("Estimated Insurance Charges")

        st.metric(
            label="Predicted Insurance Cost",
            value=f"${prediction:,.2f}"
        )

        st.info(
            "This is an estimated prediction based on "
            "the trained Machine Learning model."
        )

    except Exception as e:
        st.error(f"Prediction Error: {e}")
