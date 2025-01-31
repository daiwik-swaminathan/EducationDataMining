import pandas as pd

# Load datasets
main_table = pd.read_csv("Data/MainTable.csv")
subject_data = pd.read_csv("Data/LinkTables/Subject.csv")
early_data = pd.read_csv("Data/early.csv")
late_data = pd.read_csv("Data/late.csv")

# Print column names to verify
print("Early Data Columns:", early_data.columns)
print("Late Data Columns:", late_data.columns)

# Merge X-Grade with submission data
grades = subject_data[["SubjectID", "X-Grade"]]
main_table = main_table.merge(grades, on="SubjectID", how="left")

# Group early_data by SubjectID and calculate aggregates
# Adjust column names based on actual data
early_aggregates = early_data.groupby('SubjectID').agg({
    'Attempts': 'mean',  # Assuming 'Attempts' is a column in early_data
    'CorrectEventually': 'mean'  # Using 'CorrectEventually' as a success metric
}).reset_index()

# Group late_data by SubjectID and calculate aggregates
late_aggregates = late_data.groupby('SubjectID').agg({
    'Attempts': 'mean',  # Assuming 'Attempts' is a column in late_data
    'CorrectEventually': 'mean'  # Using 'CorrectEventually' as a success metric
}).reset_index()

# Merge aggregates with subject_data
subject_data = subject_data.merge(early_aggregates, on='SubjectID', how='left', suffixes=('', '_early'))
subject_data = subject_data.merge(late_aggregates, on='SubjectID', how='left', suffixes=('', '_late'))

# Now subject_data contains X-Grade, early and late aggregates
print(subject_data.head())

# Save the updated subject_data
subject_data.to_csv("Data/LinkTables/Subject_with_aggregates.csv", index=False)

