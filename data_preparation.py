import pandas as pd
import numpy as np
import requests
import os
from sklearn.model_selection import train_test_split
import zipfile
import io

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)

def download_dataset():
    """
    Download phishing dataset from UCI ML Repository
    """
    print("Downloading phishing dataset...")
    
    # URL for the UCI phishing dataset
    phishing_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
    
    try:
        response = requests.get(phishing_url)
        if response.status_code == 200:
            # Parse ARFF file content
            content = response.text
            data_lines = []
            header_ended = False
            
            for line in content.splitlines():
                if line.lower().startswith('@data'):
                    header_ended = True
                    continue
                
                if header_ended and line.strip() and not line.startswith('%'):
                    data_lines.append(line)
            
            # Convert to CSV format
            csv_content = '\n'.join(data_lines)
            
            # Create DataFrame
            df = pd.read_csv(io.StringIO(csv_content), header=None)
            
            # The last column contains the class: -1 (phishing) or 1 (legitimate)
            # Convert to 1 (phishing) or 0 (legitimate) for consistency
            df.iloc[:, -1] = df.iloc[:, -1].apply(lambda x: 1 if x == -1 else 0)
            
            print(f"Dataset downloaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Save the raw dataset
            df.to_csv('data/phishing_dataset_raw.csv', index=False)
            print("Raw dataset saved to data/phishing_dataset_raw.csv")
            
            return df
        else:
            print(f"Failed to download dataset: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        
        # Alternative: Use a simpler dataset from PhishTank and OpenPhish
        print("Trying alternative dataset...")
        try:
            # URL for a simpler phishing dataset
            alt_url = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/phishing-links-ACTIVE.txt"
            response = requests.get(alt_url)
            
            if response.status_code == 200:
                phishing_urls = [url.strip() for url in response.text.splitlines() if url.strip()]
                
                # Create a simple dataset with just URLs and labels
                phishing_df = pd.DataFrame({
                    'url': phishing_urls,
                    'is_phishing': 1  # All are phishing
                })
                
                # Add some legitimate URLs
                legitimate_domains = [
                    "google.com", "facebook.com", "youtube.com", "amazon.com", 
                    "wikipedia.org", "twitter.com", "instagram.com", "linkedin.com",
                    "microsoft.com", "apple.com", "netflix.com", "github.com",
                    "yahoo.com", "reddit.com", "cnn.com", "bbc.com"
                ]
                
                legitimate_urls = [f"https://www.{domain}" for domain in legitimate_domains]
                legitimate_df = pd.DataFrame({
                    'url': legitimate_urls,
                    'is_phishing': 0  # All are legitimate
                })
                
                # Combine datasets
                df = pd.concat([phishing_df.sample(min(1000, len(phishing_df))), 
                               legitimate_df], ignore_index=True)
                
                print(f"Alternative dataset created with {df.shape[0]} rows")
                
                # Save the raw dataset
                df.to_csv('data/phishing_dataset_raw.csv', index=False)
                print("Raw dataset saved to data/phishing_dataset_raw.csv")
                
                return df
            else:
                print("Failed to download alternative dataset")
                return None
        except Exception as e:
            print(f"Error downloading alternative dataset: {e}")
            print("Please download a phishing dataset manually and place it in the 'data' folder")
            return None

def prepare_dataset(df=None):
    """
    Prepare the dataset for training
    """
    if df is None:
        # Try to load the dataset if it exists
        try:
            df = pd.read_csv('data/phishing_dataset_raw.csv')
        except:
            print("Raw dataset not found. Downloading...")
            df = download_dataset()
            if df is None:
                return None
    
    print("Preparing dataset...")
    
    # Check if we have the URL-only dataset or the feature-based dataset
    if 'url' in df.columns:
        # We have the URL-only dataset, need to extract features
        from feature_extraction import extract_features_from_urls
        
        print("Extracting features from URLs...")
        # This function will be defined in feature_extraction.py
        features_df = extract_features_from_urls(df['url'].tolist())
        
        # Combine features with labels
        df = pd.concat([features_df, df['is_phishing']], axis=1)
    
    # The dataset has features with the last column as the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save the splits
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv('data/train_data.csv', index=False)
    test_data.to_csv('data/test_data.csv', index=False)
    
    print(f"Dataset prepared and split into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Download and prepare the dataset
    prepare_dataset() 