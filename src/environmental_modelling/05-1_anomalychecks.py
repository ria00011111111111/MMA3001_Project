import pandas as pd
from pathlib import Path

path = Path("outputs/modeldevelopment")

for filename in [
    "environmental_anomaly_candidates.csv",
    "anomaly_detection_summary.csv",
    "model_ready_sensor_data.csv",
]:
    file_path = path / filename

    if file_path.exists():
        df = pd.read_csv(file_path)
        print(f"\n{filename}")
        print("Rows:", len(df))
        print("Columns:", df.columns.tolist())

        if "environmental_anomaly_flag" in df.columns:
            print(df["environmental_anomaly_flag"].value_counts(dropna=False))