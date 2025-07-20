from models import models
import utils

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV

def main():
    train_data = pd.read_excel('data/train_set.xlsx')
    test_data = pd.read_excel('data/test_set.xlsx')

    df = preprocess(train_data)

    X = df.drop('Price', axis=1)
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for model_name, info in models.items():
        print(f"\nTreinando: {model_name}")
        with mlflow.start_run(run_name=model_name):
            search = GridSearchCV(
                estimator=info['model'],
                param_grid=info['params'],
                cv=3,
                n_jobs=-1,
                scoring='neg_root_mean_squared_error',
            )

            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            best_params = search.best_params_

            y_pred = best_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            print(f"R2: {r2:.3f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")

            mlflow.log_param("model_name", model_name)
            mlflow.log_params(best_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(best_model, f"name={model_name.lower()}")


def preprocess(dataframe):
    df = dataframe.copy()
    df.dropna(inplace=True)

    df['Journey_Day'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True, errors='coerce').dt.day
    df['Journey_Month'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True, errors='coerce').dt.month
    df.drop('Date_of_Journey', axis=1, inplace=True)

    df['Dep_Hour'] = pd.to_datetime(df['Dep_Time'], format='%H:%M').dt.hour
    df['Dep_Minute'] = pd.to_datetime(df['Dep_Time'], format='%H:%M').dt.minute
    df.drop('Dep_Time', axis=1, inplace=True)

    df['Arrival_Hour'] = pd.to_datetime(df['Arrival_Time'], errors='coerce').dt.hour
    df['Arrival_Minute'] = pd.to_datetime(df['Arrival_Time'], errors='coerce').dt.minute
    df.drop('Arrival_Time', axis=1, inplace=True)

    df['Duration'] = df['Duration'].apply(utils.convert_duration)

    stop_map = {
        'non-stop': 0,
        '1 stop': 1,
        '2 stops': 2,
        '3 stops': 3,
        '4 stops': 4
    }

    df['Total_Stops'] = df['Total_Stops'].map(stop_map).fillna(-1).astype(int)

    df['no_meal'] = df['Additional_Info'].apply(lambda x: 1 if x == 'In-flight meal not included' else 0)
    df['no_baggage'] = df['Additional_Info'].apply(lambda x: 1 if x == 'No check-in baggage included' else 0)
    df['info_missing'] = df['Additional_Info'].apply(lambda x: 1 if x == 'No info' else 0)

    df.drop(['Route', 'Additional_Info'], axis=1, inplace=True)

    df = pd.get_dummies(df, drop_first=True)

    return df

if __name__ == "__main__":
    main()
