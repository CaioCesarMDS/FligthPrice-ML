from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


models = {
    "LinearRegression": {
        "model": LinearRegression(),
        "params": {
            "fit_intercept": [True, False],
            "n_jobs": [None, -1, 3, 5, 10],
        }
    },
    "KNN": {
        "model": Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor())
        ]),
        "params": {
            "knn__n_neighbors": [3, 5, 7, 10, 15],
            "knn__weights": ["uniform", "distance"],
            "knn__algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "knn__leaf_size": [10, 20, 30, 40, 50]
        }
    },
    "DecisionTree": {
        "model": DecisionTreeRegressor(random_state=42),
        "params": {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 8, 10, 15],
            "min_samples_leaf": [1, 2, 4, 6, 8],
            "max_leaf_nodes": [None, 10, 20, 30, 40]
        }
    },
    "RandomForest": {
        "model": RandomForestRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100, 150, 200, 250],
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 8, 10, 15],
            "min_samples_leaf": [1, 2, 4, 6, 8]
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingRegressor(random_state=42),
        "params": {
            "n_estimators": [50, 100, 150, 200],
            "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.5],
            "max_depth": [3, 5, 7, 9 , 12],
            "min_samples_split": [2, 5, 10, 15, 20],
        }
    },
    "HistGradientBoosting": {
    "model": HistGradientBoostingRegressor(random_state=42),
    "params": {
        "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.5],
        "max_iter": [50, 100, 200, 300, 400],
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_leaf": [10, 20, 30, 40, 50],
        "l2_regularization": [0.0, 0.1, 0.5, 1.0, 2.0],
    }
}
}

