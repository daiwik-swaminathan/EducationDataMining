
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Question 1

# Open the main file
df = pd.read_csv('MainTable.csv')

# Unique number of Subject IDs in this file
print(df['SubjectID'].nunique()) # Outputs 413

# Open the subjects file
df2 = pd.read_csv('Subject.csv')

# Unique number of Subject IDs in this file
print(df2['SubjectID'].nunique()) # Outputs 372

# Question 2

# Open the two CSV files
early = pd.read_csv("early.csv") 
late = pd.read_csv("late.csv")  

# Combine the two datasets
combined = pd.concat([early, late])

# Group by ProblemID and SubjectID to find the number of attempts per student for each problem
grouped = combined.groupby(['AssignmentID', 'ProblemID', 'SubjectID'])['Attempts'].mean()

# Find the average number of attempts for each ProblemID across all students
problem_avg_attempts = grouped.groupby(['AssignmentID', 'ProblemID']).mean()

print(problem_avg_attempts)

# Question 3

# Open file
df = pd.read_csv("MainTable.csv")

# Filter for rows where Compile.Result is "error"
errors = df[df["Compile.Result"] == "Error"]

# Group by ProblemID and count errors per student
error_counts = errors.groupby(["AssignmentID", "ProblemID", "SubjectID"]).size().reset_index(name="ErrorCount")

# Calculate the average number of errors for each problem
avg_errors_per_problem = error_counts.groupby(["AssignmentID", "ProblemID"])["ErrorCount"].mean()

print(avg_errors_per_problem)

# PART 2: OPEN-ENDED ANALYSIS

# Student Struggle Across All Problems

# Open file
main_table = pd.read_csv("MainTable.csv")

# Filter for rows where there was compiler error
errors = main_table[main_table["Compile.Result"] == "Error"]

# Group by AssignmentID, ProblemID, and SubjectID to count errors per student
error_counts = errors.groupby(["AssignmentID", "ProblemID", "SubjectID"]).size().reset_index(name="ErrorCount")

# Calculate the average number of errors per problem
avg_errors_per_problem = error_counts.groupby(["AssignmentID", "ProblemID"])["ErrorCount"].mean().reset_index()

# Calculate the average number of submissions per problem per student
total_submissions = main_table.groupby(["AssignmentID", "ProblemID", "SubjectID"]).size().reset_index(name="SubmissionCount")
avg_submissions_per_problem = total_submissions.groupby(["AssignmentID", "ProblemID"])['SubmissionCount'].mean().reset_index()

# Merge average submissions and average errors
data_merged = pd.merge(avg_submissions_per_problem, avg_errors_per_problem, on=["AssignmentID", "ProblemID"], how="left").fillna(0)

# Get data for plotting
assignments = data_merged["AssignmentID"].astype(str) + "-" + data_merged["ProblemID"].astype(str)
avg_submissions = data_merged["SubmissionCount"]
avg_errors = data_merged["ErrorCount"]

# Plot stacked bar chart
plt.figure(figsize=(12, 6))
bar_width = 0.5
plt.bar(assignments, avg_submissions, color="blue", width=bar_width, label="Average Submissions")
plt.bar(assignments, avg_errors, color="red", width=bar_width, bottom=avg_submissions, label="Average Compiler Errors")

plt.xlabel("Assignment-Problem ID")
plt.ylabel("Average Count")
plt.title("Average Compiler Errors and Submissions per Problem (Stacked)")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()

# Submission Time + Box Plots of Score Distribution

# Open file
main_table = pd.read_csv("MainTable.csv")

# Convert timestamps to datetime
main_table["ServerTimestamp"] = pd.to_datetime(main_table["ServerTimestamp"])

# Filter for only Run.Program events
run_events = main_table[main_table["EventType"] == "Run.Program"].copy()

# Get hour of submission
run_events["Hour"] = run_events["ServerTimestamp"].dt.hour

# Make bar plot
plt.figure(figsize=(12, 6))
sns.barplot(x=run_events["Hour"].value_counts().sort_index().index, 
            y=run_events["Hour"].value_counts().sort_index().values, 
            color="blue")  
plt.xticks(range(24))
plt.xlabel("Hour of Day")
plt.ylabel("Total Submissions")
plt.title("Submissions by Hour of Day")
plt.show()

# Make box plot distribution chart
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=run_events, 
    x="Hour", 
    y="Score",
    boxprops={'color': 'black'},     
    whiskerprops={'color': 'black'}, 
    capprops={'color': 'black'},    
    flierprops={'marker': 'o', 'color': 'black', 'alpha': 0.5},  
    medianprops={'color': 'red', 'linewidth': 2.5}  
)
plt.xticks(range(24))
plt.xlabel("Hour of Day")
plt.ylabel("Score")
plt.title("Score Distribution by Hour of Submission")
plt.show()



