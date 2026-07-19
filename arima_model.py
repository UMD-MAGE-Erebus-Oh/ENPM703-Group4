"""
ARIMA baseline model
"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

from data_utils import get_full_flu_dataset
from eval_utils import calc_rmse

class ArimaModel:
    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.results = None

    def train(self, y: np.ndarray):
        y = np.asarray(y, dtype=float)

        self.results = ARIMA(
            y,
            order=self.order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit()

        return self
    
    def predict_in_sample(self) -> np.ndarray:
        if self.results is None:
            raise ValueError("Model must be trained first")
        
        n_obs = int(self.results.nobs)
        y_pred = self.results.predict(start=0, end=n_obs - 1)

        return np.asarray(y_pred, dtype=float)
    
    def forecast(self, steps: int) -> np.ndarray:
        if self.results is None:
            raise ValueError("Model must be trained first")
        
        y_pred = self.results.forecast(steps=steps)

        return np.asarray(y_pred, dtype=float)
    
def main():
    print("ARIMA Baseline ---")
    datasets = get_full_flu_dataset()

    y_train = datasets["y_train"]
    y_val = datasets["y_val"]
    y_test = datasets["y_test"]

    arima_order = (1, 1, 1)

    print(f"\tArima{arima_order}")

    train_model = ArimaModel(order=arima_order).train(y_train)
    y_train_pred = train_model.predict_in_sample()
    train_error = calc_rmse(y_train_pred, y_train)
    print(f"\t\ttrain error = {train_error}")

    y_val_pred = train_model.forecast(len(y_val))
    val_error = calc_rmse(y_val_pred, y_val)
    print(f"\t\tval error = {val_error}")

    test_fit_y = np.concatenate([y_train, y_val])
    test_model = ArimaModel(order=arima_order).train(test_fit_y)
    y_test_pred = test_model.forecast(len(y_test))
    test_error = calc_rmse(y_test_pred, y_test)
    print(f"\t\ttest error = {test_error}")


if __name__ == "__main__":
    main()