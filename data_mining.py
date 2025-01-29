import pandas as pd

# Question 1

df = pd.read_csv('MainTable.csv')

print(df['SubjectID'].nunique()) # Outputs 413

df2 = pd.read_csv('Subject.csv')

print(df2['SubjectID'].nunique()) # Outputs 372

# Question 2

# Load the two CSV files
early = pd.read_csv("early.csv") 
late = pd.read_csv("late.csv")  

# Combine the two datasets
combined = pd.concat([early, late])

# Group by ProblemID and SubjectID to calculate the number of attempts per student for each problem
grouped = combined.groupby(['AssignmentID', 'ProblemID', 'SubjectID'])['Attempts'].mean()

# Find the average number of attempts for each ProblemID across all students
problem_avg_attempts = grouped.groupby(['AssignmentID', 'ProblemID']).mean()

print(problem_avg_attempts)

# Question 3

# Step 1: Load the CSV file into a Pandas dataframe
df = pd.read_csv("MainTable.csv")

# Step 2: Filter for rows where Compile.Result is "error"
errors = df[df["Compile.Result"] == "Error"]

# Step 3: Group by ProblemID and count errors per student
error_counts = errors.groupby(["AssignmentID", "ProblemID", "SubjectID"]).size().reset_index(name="ErrorCount")

# print(error_counts)

# Step 4: Calculate the average number of errors for each problem
avg_errors_per_problem = error_counts.groupby(["AssignmentID", "ProblemID"])["ErrorCount"].mean()

print(avg_errors_per_problem)