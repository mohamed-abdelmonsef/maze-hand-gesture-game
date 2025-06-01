from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import logging

def train_XGBoost(X_train, y_train):
    logging.info("Training XGBoost model...")
    xgb_model = XGBClassifier(subsample= 0.8, n_estimators= 100, max_depth= 3, learning_rate= 0.3, colsample_bytree= 0.8)
    model = xgb_model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    logging.info("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model = rf_model.fit(X_train, y_train)
    return model

def train_svc(X_train, y_train):
    logging.info("Training SVC model...")
    svc_model = SVC(tol= 0.001, shrinking= False, max_iter= 1000, kernel= "poly", gamma= "scale", degree= 5, coef0= 0.5, C= 1)
    model = svc_model.fit(X_train, y_train)
    return model
