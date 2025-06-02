import mlflow
import mlflow.data
import mlflow.models
import mlflow.sklearn
import joblib
from pathlib import Path
from .evaluation import evaluate_model
import logging



def setup_mlflow_experiment(experiment_name: str, tracking_uri: str = "http://localhost:5000") -> str:
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.set_experiment(experiment_name)
    return exp.experiment_id


def log_model_with_mlflow(model, target_encoder, X_test, y_test, model_name: str, exp_id: str, output_dir: Path):
    with mlflow.start_run(experiment_id=exp_id, run_name=model_name) as run:
        logging.info(f"Logging {model_name} to MLflow...")
        mlflow.set_tag("model", model_name)

        conf_mat_path, report, accuracy = evaluate_model(X_test, y_test, model, model_name, target_encoder, output_dir)

        mlflow.log_params(model.get_params())
        mlflow.log_text(report, artifact_file=f"{model_name}_classification_report.txt")

        mlflow.log_metrics({
            "Accuracy": accuracy
        })

        mlflow.log_artifact(conf_mat_path, artifact_path="plots")


        pd_dataset = mlflow.data.from_pandas(X_test, name="Testing Dataset")
        mlflow.log_input(pd_dataset, context="Testing")

        # Log the transformer as an artifact
        mlflow.log_artifact(str(output_dir / "transformer.pkl"), artifact_path="transformer")
        mlflow.log_artifact(str(output_dir / "target_encoder.pkl"), artifact_path="target_encoder")

        joblib.dump(model, output_dir / f"{model_name}.pkl")

        signature = mlflow.models.infer_signature(X_test, y_test)
        mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=X_test.iloc[[0]])

  