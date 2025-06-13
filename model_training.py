import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Load the dataset
print("Loading dataset...")
try:
    # Try to load preprocessed data if available
    X_train = pd.read_csv('data/train_data.csv')
    X_test = pd.read_csv('data/test_data.csv')
    
    # Print column names to debug
    print("Available columns in training data:")
    print(X_train.columns.tolist())
    
    # Try to identify the target column (it might have a different name)
    target_column = None
    possible_target_names = ['is_phishing', 'phishing', 'label', 'target', 'class', 'is_malicious']
    
    for col in possible_target_names:
        if col in X_train.columns:
            target_column = col
            print(f"Found target column: '{target_column}'")
            break
    
    if target_column is None:
        # If no known target column is found, try to identify it based on binary values
        binary_cols = []
        for col in X_train.columns:
            unique_vals = X_train[col].unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                binary_cols.append(col)
        
        if binary_cols:
            # Use the last binary column as target (often the target is the last column)
            target_column = binary_cols[-1]
            print(f"Using binary column as target: '{target_column}'")
        else:
            raise ValueError("Could not identify target column. Please check your dataset.")
    
    # Separate features and target
    y_train = X_train[target_column]
    y_test = X_test[target_column]
    X_train = X_train.drop(target_column, axis=1)
    X_test = X_test.drop(target_column, axis=1)
    
    print(f"Loaded preprocessed data: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
except FileNotFoundError:
    print("Preprocessed data not found. Loading raw data...")
    try:
        # Try to load raw data and prepare it
        raw_data = pd.read_csv('data/phishing_dataset_raw.csv')
        print("Raw data loaded successfully.")
        
        # Print column names to debug
        print("Available columns in raw data:")
        print(raw_data.columns.tolist())
        
        # Try to identify the target column
        target_column = None
        possible_target_names = ['is_phishing', 'phishing', 'label', 'target', 'class', 'is_malicious']
        
        for col in possible_target_names:
            if col in raw_data.columns:
                target_column = col
                print(f"Found target column: '{target_column}'")
                break
        
        if target_column is None:
            # If no known target column is found, try to identify it based on binary values
            binary_cols = []
            for col in raw_data.columns:
                unique_vals = raw_data[col].unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                    binary_cols.append(col)
            
            if binary_cols:
                # Use the last binary column as target
                target_column = binary_cols[-1]
                print(f"Using binary column as target: '{target_column}'")
            else:
                raise ValueError("Could not identify target column. Please check your dataset.")
        
        # Split the data
        X = raw_data.drop(target_column, axis=1)
        y = raw_data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Save the preprocessed data
        os.makedirs('data', exist_ok=True)
        X_train_save = X_train.copy()
        X_train_save[target_column] = y_train
        X_train_save.to_csv('data/train_data.csv', index=False)
        
        X_test_save = X_test.copy()
        X_test_save[target_column] = y_test
        X_test_save.to_csv('data/test_data.csv', index=False)
        
        print(f"Preprocessed data saved to data/train_data.csv and data/test_data.csv")
    except FileNotFoundError:
        print("Raw data not found. Please make sure the dataset file exists.")
        exit(1)
    except Exception as e:
        print(f"Error processing raw data: {e}")
        exit(1)

# Check for class imbalance
print("\nClass distribution in training set:")
print(y_train.value_counts())

# Apply SMOTE for handling class imbalance
print("\nApplying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Create a pipeline with preprocessing and SVM
print("\nCreating SVM model pipeline...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # SVM requires scaling
    ('svm', SVC(probability=True, random_state=42))
])

# Define hyperparameters for grid search
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.1, 0.01],
    'svm__kernel': ['rbf', 'linear']
}

# Perform grid search to find the best hyperparameters
print("\nPerforming grid search for hyperparameter tuning...")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Train the model
print("\nTraining the model...")
grid_search.fit(X_train_resampled, y_train_resampled)

# Get the best model
best_model = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")

# Make predictions on the test set
print("\nEvaluating model on test set...")
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"\nTest set metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")

# Save the metrics to a file
with open('models/evaluation_metrics.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"AUC: {auc:.4f}\n")

# Generate and save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Legitimate', 'Phishing'],
            yticklabels=['Legitimate', 'Phishing'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('plots/confusion_matrix.png')
plt.close()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

# For SVM, we can't directly get feature importance like Random Forest
# Instead, we can use the absolute values of the coefficients for linear kernel
# or use permutation importance for non-linear kernels
if grid_search.best_params_['svm__kernel'] == 'linear':
    # For linear kernel, use coefficients
    feature_importance = np.abs(best_model.named_steps['svm'].coef_[0])
    feature_names = X_train.columns
    
    # Create DataFrame for feature importance
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Save feature importance to CSV
    importance_df.to_csv('plots/feature_importance.csv', index=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title('Top 20 Feature Importance (SVM Linear Kernel)')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()
else:
    # For non-linear kernels, we can't directly get feature importance
    # We'll note this in the README
    print("\nNote: Feature importance visualization is not available for non-linear SVM kernels.")
    
    # Create a placeholder file explaining this
    with open('plots/feature_importance.csv', 'w') as f:
        f.write("Feature,Importance\n")
        f.write("Note: Feature importance is not directly available for non-linear SVM kernels.\n")
    
    # Create a placeholder image
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, "Feature importance visualization is not available for non-linear SVM kernels.", 
             horizontalalignment='center', verticalalignment='center', fontsize=12)
    plt.axis('off')
    plt.savefig('plots/feature_importance.png')
    plt.close()

# Save the model
print("\nSaving the model...")
joblib.dump(best_model, 'models/phishing_detector.pkl')

print("\nModel training complete!")
print("Model saved to: models/phishing_detector.pkl")
print("Evaluation metrics saved to: models/evaluation_metrics.txt")
print("Confusion matrix saved to: plots/confusion_matrix.png") 