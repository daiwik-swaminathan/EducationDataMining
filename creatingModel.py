import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

###########################################################
# 1. Data Loading
###########################################################


def load_insights_data(
    file_path="Data/LinkTables/new_subject_data_with_addl_features.csv",
):
    """Loads the enriched dataset containing all feature aggregates.

    This dataset is expected to include columns like:
    - SubjectID
    - X-Grade (the target for regression)
    - Attempts_early, CorrectEventually_early, etc.
    - Attempts_late, CorrectEventually_late, etc.
    - Avg_Score, Total_Compile_Errors, etc.

    Returns:
        pd.DataFrame: The loaded dataset with all columns needed for ML.
    """
    df = pd.read_csv(file_path)
    return df


###########################################################
# 2. Correlation Analysis
###########################################################


def plot_correlation_matrix(df):
    """Plots a correlation matrix of the numeric columns in the DataFrame.

    # Insight: Which variables correlate with each other (and with X-Grade)?
    # This tells us at a glance where strong linear relationships might exist.

    Args:
        df (pd.DataFrame): The dataset containing numeric columns.
    """
    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Compute correlation
    corr_matrix = numeric_df.corr()

    # Plot using matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(corr_matrix, cmap="coolwarm")
    fig.colorbar(cax)

    # Set x- and y-ticks
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=90)
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_yticklabels(corr_matrix.columns)

    plt.title("Correlation Matrix", pad=20)
    plt.tight_layout()
    plt.show()


###########################################################
# 3. Model Training
###########################################################


def train_and_evaluate_model(df):
    """Trains and evaluates a Random Forest Regressor to predict X-Grade.

    # Insight: How well do the features predict final grades?
    # Look at MSE and R^2 to understand the model's performance.

    Args:
        df (pd.DataFrame): The dataset containing features and the target (X-Grade).

    Returns:
        (RandomForestRegressor, list): The trained model and the list of feature columns used.
    """
    # Drop rows with missing target or features
    df = df.dropna(subset=["X-Grade"])  # ensure we have a target

    # Example: We'll exclude non-numeric or ID columns
    feature_cols = [col for col in df.columns if col not in ["SubjectID", "X-Grade"]]

    # Further filter to numeric columns only
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df["X-Grade"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("========================================")
    print("MODEL EVALUATION")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score:         {r2:.4f}")
    print("========================================")

    plot_actual_vs_predicted(y_test, y_pred)

    # Return model and feature names for further analysis
    return model, list(X.columns)


###########################################################
# 4. Additional Visualizations for Insights
###########################################################


def plot_feature_importances(model, feature_names):
    """Plots the feature importances from the trained Random Forest model.

    # Insight: Which features are most important in predicting the final grade (X-Grade)?
    # This bar chart will show the relative importance of each feature.

    Args:
        model (RandomForestRegressor): The trained random forest model.
        feature_names (list): List of feature column names used in training.
    """
    importances = model.feature_importances_

    # Sort by importance (descending)
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


def plot_scatter_attempts_early_vs_grade(df):
    """Plots a scatter diagram of early attempts vs. final grade.

    # Insight: Does making more attempts early in the course correlate with a higher final grade?
    # This scatter plot will help us see any trend between Attempts_early and X-Grade.

    Args:
        df (pd.DataFrame): The dataset containing 'Attempts_early' and 'X-Grade'.
    """
    # Drop missing values for relevant columns
    df_plot = df.dropna(subset=["Attempts_early", "X-Grade"])

    plt.figure(figsize=(8, 6))
    plt.scatter(
        df_plot["Attempts_early"], df_plot["X-Grade"], alpha=0.7, color="purple"
    )
    plt.title("Attempts Early vs. Final Grade")
    plt.xlabel("Attempts Early")
    plt.ylabel("X-Grade")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_compile_errors_distribution(df):
    """Plots a box plot of Total_Compile_Errors for different grade bands.

    # Insight: Do students with higher grades generally have fewer compile errors?
    # This box plot will show the distribution of compile errors for different grade ranges.

    Args:
        df (pd.DataFrame): The dataset containing 'Total_Compile_Errors' and 'X-Grade'.
    """
    # Drop rows with missing values in relevant columns
    df_plot = df.dropna(subset=["Total_Compile_Errors", "X-Grade"]).copy()

    # Create grade bands (e.g., [0-0.3], (0.3-0.6], (0.6-0.8], (0.8-1.0])
    # Adjust these bins as needed for your scale of X-Grade
    bins = [0, 0.3, 0.6, 0.8, 1.0]
    labels = ["Low (0-0.3)", "Moderate (0.3-0.6)", "Good (0.6-0.8)", "High (0.8-1.0)"]
    df_plot["GradeBand"] = pd.cut(
        df_plot["X-Grade"], bins=bins, labels=labels, include_lowest=True
    )

    # Boxplot by GradeBand
    plt.figure(figsize=(8, 6))
    df_plot.boxplot(column="Total_Compile_Errors", by="GradeBand", grid=False)
    plt.title("Total Compile Errors Distribution by Grade Band")
    plt.suptitle("")  # Remove the default boxplot title
    plt.xlabel("Grade Band")
    plt.ylabel("Total Compile Errors")
    plt.tight_layout()
    plt.show()




def plot_actual_vs_predicted(y_test, y_pred):
    """Plots actual vs predicted final grades to visualize model performance.

    Args:
        y_test (array-like): Actual X-Grade values.
        y_pred (array-like): Predicted X-Grade values.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color="blue", label="Predicted vs Actual")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 
             color="red", linestyle="dashed", linewidth=2, label="Perfect Prediction Line")
    plt.xlabel("Actual X-Grade")
    plt.ylabel("Predicted X-Grade")
    plt.title("Actual vs Predicted Final Grades")
    plt.legend()
    plt.grid(True)
    plt.show()


###########################################################
# 5. Main Execution
###########################################################


def main():
    # 1. Load the new data table with aggregated features
    df = load_insights_data()

    # 2. Plot Correlation Matrix
    plot_correlation_matrix(df)

    # 3. Train and Evaluate an ML Model
    model, feature_cols = train_and_evaluate_model(df)

    # 4. Plot Feature Importances
    plot_feature_importances(model, feature_cols)

    # 5. Scatter Plot: Attempts Early vs Grade
    plot_scatter_attempts_early_vs_grade(df)

    # 6. Box Plot: Compile Errors Distribution by Grade Band
    plot_compile_errors_distribution(df)


if __name__ == "__main__":
    main()
