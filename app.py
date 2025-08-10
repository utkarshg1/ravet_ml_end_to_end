import streamlit as st
from inference import predict_species, load_model

# Load the model first
model = load_model()

# Intitialize streamlit
st.set_page_config(page_title="Iris Project")

# Add title to the page
st.title("Iris project")
st.subheader("by Utkarsh Gaikwad")

# Take input from users
sep_len = st.number_input("Sepal Length", min_value=0.00, step=0.01)
sep_wid = st.number_input("Sepal Width", min_value=0.00, step=0.01)
pet_len = st.number_input("Petal Length", min_value=0.00, step=0.01)
pet_wid = st.number_input("Petal Width", min_value=0.00, step=0.01)

# Create a button to predict results
button = st.button("Predict", type="primary")

# if button is pressed
if button:
    pred, prob_df = predict_species(model, sep_len, sep_wid, pet_len, pet_wid)
    st.subheader(f"Prediction : {pred}")
    st.subheader("Probabilities : ")
    st.dataframe(prob_df)
    st.bar_chart(prob_df.T)
