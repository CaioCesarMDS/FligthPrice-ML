from models.model_config import models
from utils.preprocessing import preprocess
from training.train_models import train_models

import pandas as pd

from sklearn.model_selection import train_test_split

def main():
    train_data = pd.read_excel('data/train_set.xlsx')
    test_data = pd.read_excel('data/test_set.xlsx')

    df = preprocess(train_data)

    X = df.drop('Price', axis=1)
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_models(models, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()

