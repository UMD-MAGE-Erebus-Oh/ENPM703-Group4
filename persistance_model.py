"""Simple persistance baseline model

aka predict future values from curent/last values
NO training
"""

import numpy as np
from data_utils import FEATURE_COLUMNS, get_full_flu_dataset
from eval_utils import calc_rmse

class PersistanceModel():


    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplemented
    

class FlatPersistanceModel(PersistanceModel):
    def predict(self, X: np.ndarray) -> np.ndarray:
        lag1_idx = FEATURE_COLUMNS.index("lag1_log")
        
        return X[:, lag1_idx].copy()

class TrendPersistanceModel(PersistanceModel):
    def predict(self, X: np.ndarray) -> np.ndarray:
        lag1_idx = FEATURE_COLUMNS.index("lag1_log")
        lag2_idx = FEATURE_COLUMNS.index("lag2_log")

        delta = X[:, lag1_idx] - X[:, lag2_idx]

        return X[:, lag1_idx] + delta



def main():
    print("Persistance Models ---")
    datasets = get_full_flu_dataset()

    for m in [FlatPersistanceModel, TrendPersistanceModel]:
        print(f"\t{m.__name__}")
        model = m()
        for s in ["train", "val", "test"]:
            X = datasets[f"X_{s}"]
            y = datasets[f"y_{s}"]

            y_pred = model.predict(X)

            # means square error
            error = calc_rmse(y_pred, y)

            print(f"\t\t{s} error = {error}")

if __name__ == "__main__":
    main()