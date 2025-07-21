from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor

models = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {
            "fit_intercept": [True, False],
        }
    },
    "KNN": {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor())
        ]),
        "params": {
            "knn__n_neighbors": [3, 5, 7, 10, 15, 20, 25],
            "knn__weights": ["uniform", "distance"],
            "knn__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "knn__leaf_size": [10, 15, 20, 30, 35, 40, 50, 60]
        }
    },
    "DecisionTree": {
        "model": DecisionTreeRegressor(random_state=42),
        "params": {
            "max_depth": [None, 5, 10, 15, 20, 25, 30],
            "min_samples_split": [2, 5, 8, 10, 15, 20, 25],
            "min_samples_leaf": [1, 2, 4, 6, 8, 10, 12, 14],
            "max_leaf_nodes": [None, 10, 20, 30, 35, 40, 50, 60]
        }
    },
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 75, 100, 150, 200, 250 ,300],
            "max_depth": [None, 5, 10, 15, 20, 25, 30],
            "min_samples_split": [2, 5, 8, 10, 15, 20, 25],
            "min_samples_leaf": [1, 2, 4, 6, 8, 10, 12, 14],
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 75, 100, 150, 200, 250, 300],
            "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.5],
            "max_depth": [1, 3, 5, 7, 9 , 12, 15],
            "min_samples_split": [2, 5, 10, 15, 20, 25, 30],
        }
    },
    "HistGradientBoosting": {
    "model": HistGradientBoostingRegressor(random_state=42),
    "params": {
        "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
        "max_iter": [50, 75, 100, 150, 200, 300, 350, 400, 450, 500],
        "max_depth": [None, 3, 5, 10, 15, 20, 25, 30],
        "min_samples_leaf": [5, 10, 20, 30, 40, 50, 60, 70, 80],
        "l2_regularization": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    }
    }
}

