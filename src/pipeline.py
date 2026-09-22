"""Preprocessing pipeline (imputation, scaling, encoding) used for training and inference."""

import pandas as pd

from src.config import RAW_DATA_PATH, RAW_DATA_SHEET


def load_data(path=RAW_DATA_PATH, sheet_name=RAW_DATA_SHEET):
    """Load the raw CobotOps sensor log: strip column names, parse timestamps, drop empty trailing columns."""
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df

