
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class LandmarkTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.Xs = ["x" + str(i) for i in range(1, 22)]
        self.Ys = ["y" + str(i) for i in range(1, 22)]

    def fit(self, X, y=None):
        return self  # No fitting necessary

    def transform(self, X, y=None):
        X_copy = X.copy()

        for index, record in X_copy.iterrows():
            # Translate
            record[self.Xs] -= record[self.Xs[0]]
            record[self.Ys] -= record[self.Ys[0]]

            # Scale
            record[self.Xs] /= (max(record[self.Xs]) - min(record[self.Xs]))
            record[self.Ys] /= (max(record[self.Ys]) - min(record[self.Ys]))

            # Rotate
            theta = np.arctan2(record[self.Xs[9]], record[self.Ys[9]])
            cos = np.cos(theta)
            sin = np.sin(theta)
            R = np.array([[cos, -sin], [sin, cos]])
            rotated_points = R @ np.vstack((record[self.Xs], record[self.Ys]))
            record[self.Xs], record[self.Ys] = rotated_points[0], rotated_points[1]

            # Adjust Xs sign
            record[self.Xs] = np.sign(record[self.Xs[5]] - record[self.Xs[9]]) * record[self.Xs]

        return X_copy.reset_index(drop=True)





def preprocess_train(df, label_encoder_path=None):
    # Encode labels
    target_encoder = LabelEncoder()
    df['label'] = target_encoder.fit_transform(df["label"])

    # Split features and labels
    X = df.drop(columns=["label"])
    y = df["label"]

    # Split before transforming
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Fit and transform on training data
    transformer = LandmarkTransformer()
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)  # Only transform

    return transformer, target_encoder, X_train_transformed, X_test_transformed, y_train, y_test


