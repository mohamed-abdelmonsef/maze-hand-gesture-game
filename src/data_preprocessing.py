
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np




def transform_record(record):

    Xs=["x"+str(i) for i in range(1,22)]
    Ys=["y"+str(i) for i in range(1,22)]

    # Translate
    record[Xs] -= record[Xs[0]]
    record[Ys] -= record[Ys[0]]

    # Scale
    record[Xs] /= (max(record[Xs]) - min(record[Xs]))
    record[Ys] /= (max(record[Ys]) - min(record[Ys]))

    # Rotate
    theta = np.arctan2(record[Xs[9]], record[Ys[9]])
    cos = np.cos(theta)
    sin = np.sin(theta)

    R = np.array([
        [cos, -sin],
        [sin, cos]
    ])

    rotated_points = R @ np.vstack((record[Xs], record[Ys]))
    record[Xs], record[Ys] = rotated_points[0], rotated_points[1]

    # Adjust Xs sign based on the difference
    record[Xs] = np.sign(record[Xs[5]] - record[Xs[9]]) * record[Xs]

    return record





def preprocess_train(df, label_encoder_path):

    df = df.apply(transform_record, axis=1)

    # Load target_encoder
    target_encoder = LabelEncoder()
    df["target"]=target_encoder.fit_transform(df["label"])


    X=df.drop(columns=["label","target"])
    y=df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df["target"])


    return target_encoder,X_train, X_test, y_train, y_test

