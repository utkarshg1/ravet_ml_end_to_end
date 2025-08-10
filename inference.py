import joblib
import pandas as pd
import streamlit as st
from loguru import logger
from constants import MODEL_FILE


@st.cache_resource
def load_model(path=MODEL_FILE):
    logger.info(f"Loading model from : {path}")
    return joblib.load(path)


def predict_species(model, sep_len, sep_wid, pet_len, pet_wid):
    data = [
        {
            "sepal_length": sep_len,
            "sepal_width": sep_wid,
            "petal_length": pet_len,
            "petal_width": pet_wid,
        }
    ]
    logger.info(f"Performing inference on {data}")
    xnew = pd.DataFrame(data)
    preds = model.predict(xnew)[0]
    logger.info(f"Predicted species : {preds}")

    probs = model.predict_proba(xnew).round(4)
    probs_df = pd.DataFrame(probs, columns=model.classes_)
    logger.info(f"Probabilities :\n{probs_df}")
    return preds, probs_df


if __name__ == "__main__":
    model = load_model()
    preds, probs = predict_species(model, sep_len=4, sep_wid=3, pet_len=2, pet_wid=1)
