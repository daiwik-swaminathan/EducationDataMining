import pandas as pd
import numpy as np


def load_data():
    """Load datasets from CSV files."""
    main_table = pd.read_csv("Data/MainTable.csv")
    subject_data = pd.read_csv("Data/LinkTables/Subject.csv")
    early_data = pd.read_csv("Data/early.csv")
    late_data = pd.read_csv("Data/late.csv")
    return main_table, subject_data, early_data, late_data


def print_column_names(early_data, late_data):
    """Print column names of early and late datasets."""
    print("Early Data Columns:", early_data.columns)
    print("Late Data Columns:", late_data.columns)


def merge_grades(main_table, subject_data):
    """Merge X-Grade with submission data."""
    grades = subject_data[["SubjectID", "X-Grade"]]
    return main_table.merge(grades, on="SubjectID", how="left")


def calculate_aggregates(data, suffix):
    """Group data by SubjectID and calculate aggregates."""
    aggregates = (
        data.groupby("SubjectID")
        .agg(
            {
                "Attempts": "mean",  # Average attempts per student
                "CorrectEventually": "mean",  # Success rate per student
            }
        )
        .reset_index()
    )
    # Rename columns to add suffix
    aggregates = aggregates.rename(
        columns=lambda x: x if x == "SubjectID" else f"{x}{suffix}"
    )
    # Debug: Print to verify structure
    print(f"Aggregates with suffix {suffix}:\n", aggregates.head())
    return aggregates


def compute_time_based_features(df):
    """
    Compute time-based features for each subject:
    - Average time gap between consecutive events
    - First vs. last timestamp (active duration)
    """
    # Convert timestamp to pandas datetime
    df["ServerTimestamp"] = pd.to_datetime(df["ServerTimestamp"], errors="coerce")

    results = []
    for subject_id, sub_df in df.groupby("SubjectID"):
        sub_df = sub_df.sort_values(by="ServerTimestamp")  # sort by time
        timestamps = sub_df["ServerTimestamp"].dropna()

        if len(timestamps) <= 1:
            avg_time_gap = np.nan
        else:
            # Compute consecutive differences in seconds (or minutes)
            diffs = timestamps.diff().dt.total_seconds()
            avg_time_gap = diffs.mean()

        # Active duration: difference between first and last
        if not timestamps.empty:
            total_duration = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
        else:
            total_duration = np.nan

        results.append(
            {
                "SubjectID": subject_id,
                "Avg_Time_Between_Events": avg_time_gap,  # in seconds
                "Total_Active_Duration": total_duration,  # in seconds
            }
        )
    return pd.DataFrame(results)


def aggregate_main_table_features(main_table):
    """
    Aggregate additional features from main_table.
    - Count of Run.Program events
    - Count of Compile events (success/fail)
    - Compile success rate
    - Avg_Score
    - Unique Problems Attempted
    """
    # We'll do multiple groupby calls or a single aggregator with custom logic.
    # Step 1: Basic aggregator
    basic_agg = (
        main_table.groupby("SubjectID")
        .agg(
            Avg_Score=("Score", "mean"),
            Total_Compile_Errors=("EventType", lambda x: (x == "Compile.Error").sum()),
            Unique_Problems_Attempted=("ProblemID", "nunique"),
            Run_Program_Count=("EventType", lambda x: (x == "Run.Program").sum()),
            Compile_Count=("EventType", lambda x: (x == "Compile").sum()),
        )
        .reset_index()
    )

    # Step 2: Compile Success Rate from the 'Compile.Result' column
    # We'll do a separate approach because we need success vs fail from 'Compile.Result'.
    compile_agg = main_table.loc[main_table["EventType"] == "Compile"]
    compile_results = (
        compile_agg.groupby("SubjectID")["Compile.Result"]
        .apply(lambda x: (x == "Success").sum() / len(x) if len(x) else np.nan)
        .reset_index(name="Compile_Success_Rate")
    )

    # Merge success rate back into basic_agg
    basic_agg = basic_agg.merge(compile_results, on="SubjectID", how="left")

    # Step 3: Time-based features
    time_df = compute_time_based_features(main_table)
    # Merge it all
    final_agg = basic_agg.merge(time_df, on="SubjectID", how="left")

    return final_agg


def merge_aggregates(subject_data, early_aggregates, late_aggregates, main_aggregates):
    """Merge early, late, and main_table aggregates with subject data."""
    # Merge early and late features
    subject_data = subject_data.merge(early_aggregates, on="SubjectID", how="left")
    subject_data = subject_data.merge(late_aggregates, on="SubjectID", how="left")
    # Merge main_table features
    subject_data = subject_data.merge(main_aggregates, on="SubjectID", how="left")
    return subject_data


def save_data(subject_data, filename):
    """Save the updated subject data to a CSV file."""
    subject_data.to_csv(filename, index=False)


def main():
    # 1. Load Datasets
    main_table, subject_data, early_data, late_data = load_data()
    print_column_names(early_data, late_data)

    # 2. Merge X-Grade
    main_table = merge_grades(main_table, subject_data)

    # 3. Compute Early and Late Aggregates
    early_aggregates = calculate_aggregates(early_data, "_early")
    late_aggregates = calculate_aggregates(late_data, "_late")

    # 4. Compute Additional Features from MainTable
    main_aggregates = aggregate_main_table_features(main_table)

    # 5. Merge All Aggregates
    subject_data = merge_aggregates(
        subject_data, early_aggregates, late_aggregates, main_aggregates
    )

    # 6. Save and/or Inspect
    print(subject_data.head(10))
    save_data(subject_data, "Data/LinkTables/new_subject_data_with_addl_features.csv")


if __name__ == "__main__":
    main()
