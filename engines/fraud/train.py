"""
Model Training Module for Fraud Detection.
Person A: Trains chosen winner model and saves artifacts to model.pkl.
"""

import os
from typing import Any


def train_fraud_model(data_path: str = "data/raw/creditcard.csv", output_path: str = "engines/fraud/model.pkl") -> Any:
    """
    Trains fraud classification model on creditcard.csv and serializes it.
    """
    print(f"Training fraud model from {data_path}...")
    # TODO: Train LightGBM/XGBoost/RandomForest with optimal strategy
    # TODO: Save model to output_path
    return None


if __name__ == "__main__":
    train_fraud_model()
