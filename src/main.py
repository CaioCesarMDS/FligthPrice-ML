from models.model_config import models
from utils.preprocessing import preprocess_base, preprocess_train
from utils.predict_and_save import predict_and_save
from training.train_models import train_models

import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split

def main():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("Flight_Price")

    train_data = pd.read_excel('data/train_set.xlsx')
    test_data = pd.read_excel('data/test_set.xlsx')

    train_df = preprocess_train(train_data)
    test_df = preprocess_base(test_data)

    X = train_df.drop('Price', axis=1)
    y = train_df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    test_df = test_df.reindex(columns=X.columns, fill_value=0)

    best_result = train_models(models, X_train, X_test, y_train, y_test)

    predict_and_save(best_result, test_df)

if __name__ == "__main__":
    main()

