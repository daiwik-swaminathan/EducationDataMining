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
    - Avg_Attempts_early, Success_Rate_early, etc.
    - Avg_Attempts_late, Success_Rate_late, etc.
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

    # Insight: This tells us how each feature is correlated with others,
    # including the final grade (X-Grade). High correlations might indicate
    # predictive power or redundancy among variables.

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

    # Insight: This model will tell us how well our features predict final grades.
    # We can look at the R^2 score and MSE to gauge predictive power.

    Args:
        df (pd.DataFrame): The dataset containing features and the target (X-Grade).
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

    # Return model in case we want to do further analysis
    return model


###########################################################
# 4. Main Execution
###########################################################


def main():
    # 1. Load the new data table with aggregated features
    df = load_insights_data()

    # 2. Plot Correlation Matrix
    plot_correlation_matrix(df)

    # 3. Train and Evaluate an ML Model
    train_and_evaluate_model(df)

    # Additional steps can be added here


if __name__ == "__main__":
    main()
