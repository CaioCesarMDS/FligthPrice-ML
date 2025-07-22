from models.model_config import models
from utils.preprocessing import preprocess_base, preprocess_train
from training.train_models import train_models

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split

def main():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Flight_Price")

    model_name = "FlightPrice"
    model_version = "latest"

    train_data = pd.read_excel('data/train_set.xlsx')
    test_data = pd.read_excel('data/test_set.xlsx')

    train_df = preprocess_train(train_data)
    test_df = preprocess_base(test_data)

    X = train_df.drop('Price', axis=1)
    y = train_df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    test_df = test_df.reindex(columns=X.columns, fill_value=0)

    # train_models(models, X_train, X_test, y_train, y_test)

    best_model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
    mlflow.sklearn.save_model(best_model, path="models/best_model")

    predictions = best_model.predict(test_df);
    test_data['Predicted_Price'] = predictions
    test_data.to_excel('data/test_set_with_predictions.xlsx', index=False)

if __name__ == "__main__":
    main()

