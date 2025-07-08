import warnings
import sys
import logging
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

# Setup logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # ✅ Step 1: Set remote tracking URI before anything else
    remote_tracking_uri = "http://ec2-51-21-202-83.eu-north-1.compute.amazonaws.com:5000/"
    mlflow.set_tracking_uri(remote_tracking_uri)

    print("Tracking URI set to:", mlflow.get_tracking_uri())

    # ✅ Step 2: Load dataset
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download training & test CSV, check your internet connection.")

    # ✅ Step 3: Prepare data
    train, test = train_test_split(data)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    # ✅ Step 4: CLI args (or defaults)
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    # ✅ Step 5: Start MLflow run
    with mlflow.start_run():
        # Model train
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)

        # Predict & evaluate
        preds = model.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, preds)

        # Print metrics
        print(f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"  RMSE: {rmse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        # ✅ Log to MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # ✅ Log model
        tracking_type = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_type != "file":
            mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(model, "model")
