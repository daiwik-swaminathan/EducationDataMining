import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load datasets
main_table = pd.read_csv("Data/MainTable.csv")
subject_data = pd.read_csv("Data/LinkTables/Subject.csv")
early_data = pd.read_csv("Data/early.csv")
late_data = pd.read_csv("Data/late.csv")

# Merge X-Grade with submission data
grades = subject_data[["SubjectID", "X-Grade"]]
main_table = main_table.merge(grades, on="SubjectID", how="left")

# Feature Engineering
# 1. Total attempts per student
attempts = main_table.groupby("SubjectID")["ProblemID"].count().reset_index()
attempts.columns = ["SubjectID", "TotalAttempts"]

# 2. Average Score per student
avg_score = main_table.groupby("SubjectID")["Score"].mean().reset_index()
avg_score.columns = ["SubjectID", "AvgScore"]

# 3. Success Rate (CorrectEventually)
early_success = (
    early_data.groupby("SubjectID")["CorrectEventually"].mean().reset_index()
)
early_success.columns = ["SubjectID", "EarlySuccessRate"]

late_success = late_data.groupby("SubjectID")["CorrectEventually"].mean().reset_index()
late_success.columns = ["SubjectID", "LateSuccessRate"]

# 4. Compile Errors Frequency
compile_errors = (
    main_table[main_table["EventType"] == "Compile.Error"]
    .groupby("SubjectID")["EventType"]
    .count()
    .reset_index()
)
compile_errors.columns = ["SubjectID", "CompileErrors"]
compile_errors["CompileErrors"] = compile_errors["CompileErrors"].fillna(0)

# 5. Time spent (difference between first and last submission)
main_table["ServerTimestamp"] = pd.to_datetime(main_table["ServerTimestamp"])
time_spent = (
    main_table.groupby("SubjectID")["ServerTimestamp"].agg(["min", "max"]).reset_index()
)
time_spent["TimeSpent"] = (time_spent["max"] - time_spent["min"]).dt.total_seconds()
time_spent = time_spent[["SubjectID", "TimeSpent"]]

# Merge all features into one dataframe
features = grades.merge(attempts, on="SubjectID", how="left")
features = features.merge(avg_score, on="SubjectID", how="left")
features = features.merge(early_success, on="SubjectID", how="left")
features = features.merge(late_success, on="SubjectID", how="left")
features = features.merge(compile_errors, on="SubjectID", how="left")
features = features.merge(time_spent, on="SubjectID", how="left")

# Drop rows with missing values
features = features.dropna()

# Train-test split
X = features.drop(columns=["SubjectID", "X-Grade"])
y = features["X-Grade"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Plot Predictions
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual X-Grade")
plt.ylabel("Predicted X-Grade")
plt.title("Actual vs Predicted Final Grades")
plt.show()
