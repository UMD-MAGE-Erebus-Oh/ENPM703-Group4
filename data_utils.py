"""Dataset loading utilities

Taken from eda.ipynb

All data under data/ folder

Raw Data:
incident_hosp.csv - FluSight incident hopsitalizations
hosp_admissions.csv - CDC NHSN confirmed hopsitalizations
google_trends.csv - Google Trends results for flu search terms

Cleaned Data:
combined_flu_us.csv - national weekly hopsitalizations
state_hosp.csv - state-level weekly hopsitalizations
flu_features.csv - weekly features and hopsital admissions
"""

from pathlib import Path
import pandas as pd

THIS_DIR = Path(__file__).parent

DATASET_FILE = THIS_DIR / "data" / "flu_features.csv"

# Oct - May
ACTIVE_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5]
COMPLETE_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

FEATURE_COLUMNS = [
    # Lag features (log scale)
    "lag1_log",
    "lag2_log",
    "lag3_log",
    "lag4_log",
    # Momentum features
    "wow_pct_lag1",
    "wow_pct_lag2",
    # Rolling statistics (log scale)
    "roll4_mean_log",
    "roll4_std_log",
    # Calendar features
    "month_sin",
    "month_cos",
    "week_sin",
    "week_cos",
    # Zone flags
    "is_onset_zone",
    "is_peak_zone",
]

TARGET_COLUMN = "target"

METADATA_COLUMNS = ["date", "season", "weekly_admissions", "roll4"]

def load_flu_features_data():
    df = pd.read_csv(DATASET_FILE, parse_dates=["date"])

    return df.sort_values("date").reset_index(drop=True)

def convert_xy(df: pd.DataFrame):
    return df[FEATURE_COLUMNS].to_numpy(), df[TARGET_COLUMN].to_numpy()


def get_full_flu_dataset() -> dict:
    df = load_flu_features_data()

    train_mask_1 = df["season"] == "2022-23"
    train_mask_2 = df["season"] == "2023-24"
    train_mask = train_mask_1 | train_mask_2
    val_mask = df["season"] == "2024-25"
    test_mask = df["season"] == "2025-26"

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    X_train, y_train = convert_xy(train_df)
    X_val, y_val = convert_xy(val_df)
    X_test, y_test = convert_xy(test_df)

    datasets = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test
    }

    return datasets