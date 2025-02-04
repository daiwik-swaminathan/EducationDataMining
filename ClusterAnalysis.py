import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_students(file_path):
    df = pd.read_csv(file_path)
    
    student_features = []
    for student_id in df['SubjectID'].unique():
        student_df = df[df['SubjectID'] == student_id]
        
        error_rate = len(student_df[student_df['EventType'] == 'Compile.Error']) / len(student_df)
        runs = len(student_df[student_df['EventType'] == 'Run.Program'])
        problems = len(student_df['ProblemID'].unique())
        
        scores = student_df[student_df['EventType'] == 'Run.Program']['Score'].fillna(0)
        
        student_features.append({
            'student_id': student_id,
            'error_rate': error_rate,
            'runs_per_problem': runs / problems if problems > 0 else 0,
            'avg_score': scores.mean() if len(scores) > 0 else 0
        })
    
    return pd.DataFrame(student_features)

def cluster_and_visualize(features_df):
    features = ['error_rate', 'runs_per_problem', 'avg_score']
    X = StandardScaler().fit_transform(features_df[features])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    features_df['cluster'] = kmeans.fit_predict(X)
    
    plt.figure(figsize=(10, 5))
    
    # Plot 1: Error Rate vs Score
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=features_df, x='error_rate', y='avg_score', 
                   hue='cluster', palette='deep')
    plt.title('Error Rate vs Average Score')
    
    # Plot 2: Runs vs Score
    plt.subplot(1, 2, 2)
    sns.scatterplot(data=features_df, x='runs_per_problem', y='avg_score', 
                   hue='cluster', palette='deep')
    plt.title('Runs per Problem vs Average Score')
    
    plt.tight_layout()
    return features_df

def main(file_path='/Users/sameernadeem/313-dataMining-Sameer/All/Data/MainTable.csv'):
    student_features = analyze_students(file_path)
    
    clustered_df = cluster_and_visualize(student_features)
    
    print("\nCluster Summary:")
    for cluster in sorted(clustered_df['cluster'].unique()):
        cluster_data = clustered_df[clustered_df['cluster'] == cluster]
        print(f"\nCluster {cluster}:")
        print(f"Number of students: {len(cluster_data)}")
        print(f"Average score: {cluster_data['avg_score'].mean():.2f}")
        print(f"Average error rate: {cluster_data['error_rate'].mean():.2f}")
        print(f"Average runs per problem: {cluster_data['runs_per_problem'].mean():.2f}")
    
    plt.show()
    return clustered_df

if __name__ == "__main__":
    main()