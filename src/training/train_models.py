import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

def train_models(models, X_train, X_test, y_train, y_test):
	results = []
	for model_name, info in models.items():
		print(f"\nTraining: {model_name}")
		with mlflow.start_run(run_name=model_name):
			search = RandomizedSearchCV(
				estimator=info['model'],
				param_distributions=info['params'],
				n_iter=150,
				cv=7,
				n_jobs=-1,
				scoring='neg_root_mean_squared_error',
				random_state=42,
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
			mlflow.sklearn.log_model(
				sk_model=best_model,
				name="model",
				input_example=X_train,
			)

			results.append({
				"model_name": model_name,
				"model": best_model,
				"params": best_params,
				"rmse": rmse,
				"mae": mae,
				"r2": r2
			})

		mlflow.end_run()

	best_result = sorted(
		results,
		key=lambda x: (x["rmse"], x["mae"], -x["r2"])
	)[0]

	return best_result
