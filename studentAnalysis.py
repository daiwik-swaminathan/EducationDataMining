import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def analyze_data(file_path):
    df = pd.read_csv(file_path)
    df['CorrectEventually'] = df['CorrectEventually'].astype(int)
    df['Label'] = df['Label'].astype(int)
    
    student_stats = df.groupby('SubjectID').agg({
        'Attempts': 'mean',
        'CorrectEventually': 'mean',
        'Label': 'mean'
    }).rename(columns={
        'Attempts': 'avg_attempts',
        'CorrectEventually': 'success_rate',
        'Label': 'struggle_rate'
    })
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=student_stats, x='avg_attempts', y='struggle_rate')
    plt.title('Average Attempts vs Struggle Rate per Student')
    plt.xlabel('Average Attempts per Problem')
    plt.ylabel('Proportion of Struggling Problems')
    
    X = df[['Attempts', 'CorrectEventually']]
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    print("\nKey Insights:")
    print(f"Total students: {len(df['SubjectID'].unique())}")
    print(f"Average attempts per problem: {df['Attempts'].mean():.2f}")
    print(f"Success rate: {df['CorrectEventually'].mean()*100:.1f}%")
    print(f"Struggle rate: {df['Label'].mean()*100:.1f}%")
    print(f"Correlation (attempts vs success): {df['Attempts'].corr(df['CorrectEventually']):.3f}")
    
    plt.show()

if __name__ == "__main__":
    analyze_data('/Users/sameernadeem/313-dataMining-Sameer/All/early.csv')