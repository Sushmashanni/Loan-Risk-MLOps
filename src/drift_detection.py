import os
import pandas as pd

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def main():

    # Load datasets
    reference_data = pd.read_csv("data/train.csv")
    current_data = pd.read_csv("data/test.csv")

    # Create report
    report = Report(metrics=[
        DataDriftPreset()
    ])

    # Run report
    report.run(
        reference_data=reference_data,
        current_data=current_data
    )

    # Create reports folder
    os.makedirs("reports", exist_ok=True)

    # Save HTML report
    report.save_html("reports/drift_report.html")

    print("\nDrift report generated successfully!")
    print("Saved to reports/drift_report.html")


if __name__ == "__main__":
    main()