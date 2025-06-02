from src.data_preprocessing import preprocess_train
from pathlib import Path
import logging
from colorama import Fore, Style
import pandas as pd
from src.models import train_XGBoost, train_random_forest, train_svc
import joblib
from src.mlflow_logging import log_model_with_mlflow, setup_mlflow_experiment



def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=f"{Fore.GREEN}%(asctime)s{Style.RESET_ALL} - {Fore.BLUE}%(levelname)s{Style.RESET_ALL} - %(message)s"
    )




def main():
    setup_logging()
    logging.info("Starting Prediction Experiment...")
    experiment_id = setup_mlflow_experiment("hand_gesture_experiment")

    BASE_DIR = Path(__file__).resolve().parent
    data_path = BASE_DIR / "raw_data" / "hand_landmarks_data.csv"
    label_encoder_path = BASE_DIR/ "preprocessing_models" / "target_encoder.pkl"
    output_dir = BASE_DIR / "output"

    df = pd.read_csv(data_path)
    
    transformer,target_encoder, X_train, X_test, y_train, y_test = preprocess_train(df, label_encoder_path)

    # Save transformer locally
    joblib.dump(target_encoder, output_dir / "target_encoder.pkl")
    # Save the transformer to a file
    joblib.dump(transformer, output_dir / "transformer.pkl")

    
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    xg_model = train_XGBoost(X_train, y_train)
    log_model_with_mlflow(xg_model, target_encoder, X_test, y_test, "XGboost", experiment_id, output_dir)

    rf_model = train_random_forest(X_train, y_train)
    log_model_with_mlflow(rf_model, target_encoder, X_test, y_test, "RandomForestClassifier", experiment_id, output_dir)

    svc_model = train_svc(X_train, y_train)
    log_model_with_mlflow(svc_model, target_encoder, X_test, y_test, "SVC_classifier", experiment_id, output_dir)


if __name__ == "__main__":
    main()
