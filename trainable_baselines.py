"""
For trainable baseline models
"""

import numpy as np

from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression, Ridge

from data_utils import get_full_flu_dataset
from eval_utils import calc_rmse

class TrainableModel(ABC):
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


class LinearRegressionModel(TrainableModel):
    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model needs to be trained first")
        
        return self.model.predict(X)
    
class RidgeRegressionModel(TrainableModel):
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.is_trained = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model needs to be trained first")
    
        return self.model.predict(X)


def evaluate_model(model: TrainableModel, datasets: dict, model_name: str):
    print(f"\t{model_name}")
    model.train(datasets["X_train"], datasets["y_train"])

    for split in ["train", "val", "test"]:
        X = datasets[f"X_{split}"]
        y = datasets[f"y_{split}"]

        y_pred = model.predict(X)
        error = calc_rmse(y_pred, y)

        print(f"\t\t{split} error = {error}")


def main():
    print("Trainablle Baseline Models ---")
    datasets = get_full_flu_dataset()

    evaluate_model(LinearRegressionModel(), datasets, "Linear Regression")
    evaluate_model(RidgeRegressionModel(), datasets, "Ridge Regression")


if __name__ == "__main__":
    main()