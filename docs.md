# Programming Education Data Analysis

This document provides an overview of the tasks we performed for the **CS 1 Programming Process** dataset assignment. It includes instructions for obtaining the data, our exploratory data analysis (Task 1), and an open-ended analysis with modeling and feature engineering (Task 2). The final section outlines how to run the code we developed.

---

## Table of Contents

1. [Dataset Description](#dataset-description)
2. [Obtaining the Data](#obtaining-the-data)
3. [Data Organization](#data-organization)
4. [Task 1: Exploratory Data Analysis](#task-1-exploratory-data-analysis)

- [Question 1](#question-1)
- [Question 2](#question-2)
- [Question 3](#question-3)

5. [Task 2: Open-Ended Analysis](#task-2-open-ended-analysis)

- [Overview of Approach](#overview-of-approach)
- [Feature Engineering](#feature-engineering)
- [Modeling & Insights](#modeling--insights)

6. [How to Run the Code](#how-to-run-the-code)
7. [References & Resources](#references--resources)

---

## Dataset Description

The dataset contains approximately 70,000 programming problem submissions (in Java) from students in a CS 1 course. Each student was allowed unlimited submissions for each problem. The dataset also includes each student's **final course grade**.

The key files in the dataset structure are:

```
Root/
   early.csv
   late.csv
   Data/
       MainTable.csv
       Metadata.csv
       CodeStates/
           CodeState.csv
       LinkTables/
           Subject.csv
```

### MainTable.csv

- **Primary Columns**:
- `SubjectID`: A unique ID for each student
- `AssignmentID`: The ID of the assignment
- `ProblemID`: The ID of the specific problem
- `CodeStateID`: A unique ID corresponding to a snapshot of a student’s code
- `EventType`: Examples include `Run.Program`, `Compile`, `Compile.Error`
- `Score`: The score (0 to 1) for `Run.Program` events
- `ServerTimestamp`: Timestamp of the event
- `ServerTimezone`: Timezone offset (0 or UTC)

### CodeState.csv

- This maps each `CodeStateID` in `MainTable.csv` to the corresponding source code text.

### early.csv and late.csv

- **early.csv**: Contains aggregated data (like `Attempts` and `CorrectEventually`) for each `(SubjectID, ProblemID)` pair for the **first 30 problems** (3 assignments).
- **late.csv**: Same structure, but for the **last 20 problems** (2 assignments).

### Subject.csv

- Contains mappings from `SubjectID` to the **final grades** (column: `X-Grade`) of students.

---

## Data Organization

Below is the folder structure that we used in our working directory:

```
Root/
   early.csv
   late.csv
   Data/
       MainTable.csv
       LinkTables/
           Subject.csv
       CodeStates/
           CodeState.csv
   ...
(Additional scripts and notebooks)
```

---

## Task 1: Exploratory Data Analysis

For Task 1, we answered three specific questions to become familiar with the data. We used Python (pandas) to process the `.csv` files.

### Question 1

**Q:** How many unique students are represented in the dataset?

**A:**

- If we use `MainTable.csv`, we see **413** unique `SubjectID`s.
- If we use `Subject.csv`, we see **372** unique `SubjectID`s.

**Why do these answers differ?**
In many datasets, some subjects may appear in **MainTable.csv** (because of submissions or other logged events) even if they eventually dropped or never received a final grade. Conversely, **Subject.csv** might only list those who have a final grade. Thus, `Subject.csv` might have fewer entries if not all who appeared in the logs remained in the class or vice versa.

**Python code snippet**:

```python
import pandas as pd


df_main = pd.read_csv("MainTable.csv")
print(df_main["SubjectID"].nunique())  # Outputs 413


df_subject = pd.read_csv("Subject.csv")
print(df_subject["SubjectID"].nunique())  # Outputs 372
```

---

### Question 2

**Q:** Which Problem had the highest average number of attempts per student?

We needed both **early.csv** and **late.csv** because the problems in the data span multiple assignments. We computed the average attempts per problem by merging these files, grouping by `(AssignmentID, ProblemID, SubjectID)`, and then averaging.

**Short Answer:**
The top problem(s) will appear in our results as having the highest average attempts per student.

**Python code snippet**:

```python
import pandas as pd


# Load data
early = pd.read_csv("early.csv")
late = pd.read_csv("late.csv")


# Combine
combined = pd.concat([early, late])


# Group by problem and subject to find average attempts
grouped = combined.groupby(["AssignmentID", "ProblemID", "SubjectID"])["Attempts"].mean()


# Then find the average across all subjects per problem
problem_avg_attempts = grouped.groupby(["AssignmentID", "ProblemID"]).mean()


# Sort descending to see which is highest
problem_avg_attempts_sorted = problem_avg_attempts.sort_values(ascending=False)
print(problem_avg_attempts_sorted.head(10))
```

_(The exact highest problem ID will appear at the top of this sorted list.)_

---

### Question 3

**Q:** On which problem did students face the most compiler errors?
We discussed two possible interpretations:

1. **Compute the average number of compiler errors** per student for each problem, then find the problem with the highest average.
2. **Compute the total number of compiler errors** across all students for each problem, then find the highest total.

Which is more meaningful depends on context. Often, the average number of errors per student (interpretation #1) provides better insight into the **per-student struggle**. The total number of errors might be skewed by the total number of attempts or by how many students actually worked on that problem.

**Sample code snippet** (average compiler errors per problem):

```python
import pandas as pd


df = pd.read_csv("MainTable.csv")


# Filter rows where compilation failed
errors = df[df["Compile.Result"] == "Error"]


# Count errors per (ProblemID, SubjectID)
error_counts = (
   errors.groupby(["AssignmentID", "ProblemID", "SubjectID"])
   .size()
   .reset_index(name="ErrorCount")
)


# Compute the average error count per problem
avg_errors_per_problem = error_counts.groupby(["AssignmentID", "ProblemID"])["ErrorCount"].mean()


# Sort descending
avg_errors_per_problem_sorted = avg_errors_per_problem.sort_values(ascending=False)
print(avg_errors_per_problem_sorted.head(10))
```

---

## Task 2: Open-Ended Analysis

### Overview of Approach

After the basic EDA, we performed additional analyses to derive more insights, focusing on:

1. **Feature Engineering**: Creating aggregated metrics per student.
2. **Modeling**: Using a Random Forest Regressor to predict the final grade (`X-Grade`).

We wanted to see if early attempts, compile errors, or time-based features correlated with final performance.

---

### Feature Engineering

In `Organize data` step, we wrote a script that:

1. **Merged `MainTable.csv` with `Subject.csv`** to attach `X-Grade`.
2. **Created Aggregates** from **early.csv** and **late.csv**:

- Average attempts per student, success rates, etc.

3. **Extracted Additional Features** from **MainTable.csv**:

- Total compile errors
- Compile success rate
- Average score on `Run.Program`
- Time-based features (e.g., average time between events, total active duration)

**Code**: `organize_data.py` (example shown below):

```python
import pandas as pd
import numpy as np


def load_data():
   """Load datasets from CSV files."""
   main_table = pd.read_csv("Data/MainTable.csv")
   subject_data = pd.read_csv("Data/LinkTables/Subject.csv")
   early_data = pd.read_csv("early.csv")
   late_data = pd.read_csv("late.csv")
   return main_table, subject_data, early_data, late_data


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
               "Attempts": "mean",          # Average attempts per student
               "CorrectEventually": "mean", # Success rate
           }
       )
       .reset_index()
   )
   # Rename columns to add suffix
   aggregates = aggregates.rename(
       columns=lambda x: x if x == "SubjectID" else f"{x}{suffix}"
   )
   return aggregates


def compute_time_based_features(df):
   """Compute time-based features for each subject."""
   df["ServerTimestamp"] = pd.to_datetime(df["ServerTimestamp"], errors="coerce")


   results = []
   for subject_id, sub_df in df.groupby("SubjectID"):
       sub_df = sub_df.sort_values(by="ServerTimestamp")
       timestamps = sub_df["ServerTimestamp"].dropna()


       if len(timestamps) <= 1:
           avg_time_gap = np.nan
       else:
           diffs = timestamps.diff().dt.total_seconds()
           avg_time_gap = diffs.mean()


       if not timestamps.empty:
           total_duration = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()
       else:
           total_duration = np.nan


       results.append(
           {
               "SubjectID": subject_id,
               "Avg_Time_Between_Events": avg_time_gap,
               "Total_Active_Duration": total_duration,
           }
       )
   return pd.DataFrame(results)


def aggregate_main_table_features(main_table):
   """Aggregate additional features from MainTable."""
   basic_agg = (
       main_table.groupby("SubjectID")
       .agg(
           Avg_Score=("Score", "mean"),
           Total_Compile_Errors=(
               "EventType", lambda x: (x == "Compile.Error").sum()
           ),
           Unique_Problems_Attempted=("ProblemID", "nunique"),
           Run_Program_Count=("EventType", lambda x: (x == "Run.Program").sum()),
           Compile_Count=("EventType", lambda x: (x == "Compile").sum()),
       )
       .reset_index()
   )


   compile_agg = main_table.loc[main_table["EventType"] == "Compile"]
   compile_results = (
       compile_agg.groupby("SubjectID")["Compile.Result"]
       .apply(lambda x: (x == "Success").sum() / len(x) if len(x) else np.nan)
       .reset_index(name="Compile_Success_Rate")
   )


   basic_agg = basic_agg.merge(compile_results, on="SubjectID", how="left")


   time_df = compute_time_based_features(main_table)
   final_agg = basic_agg.merge(time_df, on="SubjectID", how="left")


   return final_agg


def merge_aggregates(subject_data, early_aggregates, late_aggregates, main_aggregates):
   """Merge early, late, and main_table aggregates with subject data."""
   subject_data = subject_data.merge(early_aggregates, on="SubjectID", how="left")
   subject_data = subject_data.merge(late_aggregates, on="SubjectID", how="left")
   subject_data = subject_data.merge(main_aggregates, on="SubjectID", how="left")
   return subject_data


def main():
   main_table, subject_data, early_data, late_data = load_data()


   # Merge X-Grade into the main table
   main_table = merge_grades(main_table, subject_data)


   # Compute aggregates
   early_aggregates = calculate_aggregates(early_data, "_early")
   late_aggregates = calculate_aggregates(late_data, "_late")
   main_aggregates = aggregate_main_table_features(main_table)


   # Merge all aggregates
   subject_data = merge_aggregates(
       subject_data, early_aggregates, late_aggregates, main_aggregates
   )


   # Save to new CSV
   subject_data.to_csv("Data/LinkTables/new_subject_data_with_addl_features.csv", index=False)


if __name__ == "__main__":
   main()
```

---

### Modeling & Insights

We next trained a **Random Forest Regressor** to predict students’ final grades (`X-Grade`) using the engineered features. We evaluated model performance via **MSE** and **R^2**.

The relevant script: `model_insights.py`.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


def load_insights_data(file_path="Data/LinkTables/new_subject_data_with_addl_features.csv"):
   df = pd.read_csv(file_path)
   return df


def plot_correlation_matrix(df):
   numeric_df = df.select_dtypes(include=[np.number])
   corr_matrix = numeric_df.corr()


   fig, ax = plt.subplots(figsize=(10, 8))
   cax = ax.matshow(corr_matrix, cmap="coolwarm")
   fig.colorbar(cax)
   ax.set_xticks(range(len(corr_matrix.columns)))
   ax.set_xticklabels(corr_matrix.columns, rotation=90)
   ax.set_yticks(range(len(corr_matrix.columns)))
   ax.set_yticklabels(corr_matrix.columns)
   plt.title("Correlation Matrix", pad=20)
   plt.tight_layout()
   plt.show()


def train_and_evaluate_model(df):
   df = df.dropna(subset=["X-Grade"])
   feature_cols = [c for c in df.columns if c not in ["SubjectID", "X-Grade"]]


   X = df[feature_cols].select_dtypes(include=[np.number])
   y = df["X-Grade"]


   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


   model = RandomForestRegressor(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)


   mse = mean_squared_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)
   print("Mean Squared Error:", mse)
   print("R^2 Score:", r2)


   # Plot predicted vs actual
   plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
   plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linestyle="--")
   plt.xlabel("Actual X-Grade")
   plt.ylabel("Predicted X-Grade")
   plt.title("Actual vs. Predicted Final Grades")
   plt.show()


   return model, feature_cols


def plot_feature_importances(model, feature_names):
   importances = model.feature_importances_
   indices = np.argsort(importances)[::-1]
   sorted_importances = importances[indices]
   sorted_features = [feature_names[i] for i in indices]


   plt.figure(figsize=(10, 6))
   plt.barh(range(len(sorted_importances)), sorted_importances[::-1], color="skyblue")
   plt.yticks(range(len(sorted_importances)), sorted_features[::-1])
   plt.title("Feature Importances")
   plt.xlabel("Importance Score")
   plt.ylabel("Features")
   plt.tight_layout()
   plt.show()


def main():
   df = load_insights_data()
   plot_correlation_matrix(df)
   model, feature_cols = train_and_evaluate_model(df)
   plot_feature_importances(model, feature_cols)


if __name__ == "__main__":
   main()
```

**Key Insights**:

- **Correlation Matrix**: We can see how each feature correlates with `X-Grade`.
- **Random Forest Results**: The MSE and R² indicate how well these features can predict final grades.
- **Feature Importances**: Which features (e.g., average early attempts, total compile errors, etc.) have the strongest predictive value?

---

## How to Run the Code

1. **Place the `.csv` files** (as downloaded) in the following structure:

```
Root/
    Data/
        early.csv
        late.csv
        MainTable.csv
        LinkTables/
            Subject.csv
        CodeStates/
            CodeState.csv
    ...
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run `organize_data.py`** to generate the enriched dataset:

```bash
python organize_data.py
```

This will create a new CSV file: `Data/LinkTables/new_subject_data_with_addl_features.csv`.

4. **Run `creatingModel.py`** to train the Random Forest model and display plots:

```bash
python model_insights.py
```

This will output:

- Correlation matrix heatmap
- Model performance metrics (MSE, R²)
- Actual vs. predicted final grades scatter plot
- Feature importances

### Insights

#### Correlation Matrix Heatmap

![Heatmap](/insights/correlation-matrix.png)

#### Feature Importances

![](/insights//feature-importance.png)

#### Early vs Final

![](/insights/early-vs-final.png)

#### Actual vs Predicted

![](/insights/actual-vs-predicted.png)

#### Total compile errors

![](/insights/total-compile-errors.png)

---

## References & Resources

- **ProgSnap2**: Standardized data format for programming education research. [Specification Link](https://progsnap2.specs.io/)
- **Python libraries**:
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [matplotlib](https://matplotlib.org/)
- **DataShop**: [PSLC DataShop](https://pslcdatashop.web.cmu.edu/)
- **Tableau**: [Tableau Public](https://public.tableau.com/en-us/s/)

---
