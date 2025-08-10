from loguru import logger
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
import joblib

from constants import DATA_FILE, MODEL_DIR, MODEL_FILE, TEST_SIZE, RANDOM_STATE, TARGET


def train_and_save_model():
    try:
        # Step 1 - Data ingestion
        logger.info(f"Performing data ingestion on {DATA_FILE} ...")
        df = pd.read_csv(DATA_FILE)
        logger.info(f"Data shape : {df.shape}")
        logger.info(f"Data columns : {df.columns.tolist()} ")

        # Step 2 - Remove duplicates
        dup = df.duplicated().sum()
        logger.info(f"Duplicates found : {dup}")
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
        logger.info(f"Duplicates dropped , data shape : {df.shape}")

        # Check for missing values
        m = df.isna().sum()
        logger.info(f"Missing Values :{m.to_dict()}")

        # Seperate X and Y
        logger.info("Seperating X and Y")
        X = df.drop(columns=[TARGET])
        Y = df[TARGET]

        # Apply train test split
        logger.info(
            f"Applying train test split with test_size={TEST_SIZE}, random state={RANDOM_STATE}"
        )
        xtrain, xtest, ytrain, ytest = train_test_split(
            X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        logger.info(f"xtrain shape : {xtrain.shape}, ytrain shape : {ytrain.shape}")
        logger.info(f"xtest shape : {xtest.shape}, ytest shape : {ytest.shape}")

        # Intitialzie a pipeline model
        logger.info("Intitializing model pipeline ...")
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(random_state=RANDOM_STATE),
        )

        # Cross validate the model
        scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring="f1_macro")
        logger.info(f"Cross Validation scores : {scores}")
        scores_mean = scores.mean().round(4)
        scores_std = scores.std().round(4)
        logger.info(f"Mean Cross validation score : {scores_mean} +/- {scores_std}")

        # Train the model
        logger.info("Training model")
        model.fit(xtrain, ytrain)
        logger.info("Model Training Done")

        # Evaluate model
        ypred_train = model.predict(xtrain)
        ypred_test = model.predict(xtest)
        f1_train = f1_score(ytrain, ypred_train, average="macro")
        f1_test = f1_score(ytest, ypred_test, average="macro")
        logger.info(f"F1 macro Train : {f1_train:.4f}")
        logger.info(f"F1 macro Test : {f1_test:.4f}")
        logger.info(
            f"Classification Report Test :\n{classification_report(ytest, ypred_test)}"
        )

        # Save the model as joblib
        logger.info(f"Saving model object to : {MODEL_FILE}")
        MODEL_DIR.mkdir(exist_ok=True)  # Create a model directory
        joblib.dump(model, MODEL_FILE)
        logger.info(f"{MODEL_FILE} Saved successfully")

        logger.success("Training pipeline successful")

    except Exception as e:
        logger.error(f"Exception occured : {e}")


if __name__ == "__main__":
    train_and_save_model()
