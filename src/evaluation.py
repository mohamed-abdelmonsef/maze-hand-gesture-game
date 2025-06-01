import matplotlib.pyplot as plt
from pathlib import Path
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import seaborn as sns
import joblib


def evaluate_model(X, Y, model, model_name, target_encoder, output_dir: Path):
    logging.info("Calculating evaluation metrics...")

    # 5-fold cross-validation
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    loaded_target_encoder = target_encoder

    y_true_all = []
    y_pred_all = []

    for train_index, test_index in kf.split(X, Y):
        # Use iloc[] to index DataFrames by position
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        # Model training
        model.fit(X_train, y_train)
        
        # Prediction
        y_pred = model.predict(X_test)
        
        # Save output
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Confusion Matrix
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=loaded_target_encoder.classes_, yticklabels=loaded_target_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix Across 5-Folds')
    output_dir.mkdir(parents=True, exist_ok=True)
    conf_mat_path = output_dir / f"confusion_mat_{model_name}.png"
    plt.savefig(conf_mat_path)
    plt.close()

    # Classification report for overall performance
    report = classification_report(y_true_all, y_pred_all, target_names=loaded_target_encoder.classes_)

    # Mean accuracy
    accuracy = accuracy_score(y_true_all, y_pred_all)


    return conf_mat_path, report, accuracy
