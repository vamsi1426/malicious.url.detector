# Add at the top of app.py, before other imports
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extraction import extract_features_from_urls
import time
import requests
from urllib.parse import urlparse
import tldextract
import whois
from datetime import datetime
import re
import logging
from llm_integration import LLMAnalyzer

# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/phishing_detector.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set page configuration
st.set_page_config(
    page_title="URL Phishing Detector",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/phishing_detector.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load evaluation metrics
@st.cache_data
def load_metrics():
    try:
        metrics = {}
        with open('models/evaluation_metrics.txt', 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                metrics[key] = float(value)
        return metrics
    except Exception as e:
        st.warning(f"Could not load metrics: {e}")
        return {
            "Accuracy": None,
            "Precision": None,
            "Recall": None,
            "F1 Score": None,
            "AUC": None
        }

# Add this to your app.py, after loading the model
@st.cache_resource
def load_llm_analyzer():
    """Load the LLM analyzer"""
    try:
        # Check if API key is available
        if os.getenv("OPENROUTER_API_KEY"):
            return LLMAnalyzer(
                model_provider="openrouter", 
                model_name="meta-llama/llama-3.3-70b-instruct:free"
            )
        elif os.getenv("OPENAI_API_KEY"):
            return LLMAnalyzer(model_provider="openai")
        elif os.getenv("ANTHROPIC_API_KEY"):
            return LLMAnalyzer(model_provider="anthropic")
        elif os.getenv("META_API_KEY"):
            return LLMAnalyzer(model_provider="meta")
        else:
            st.warning("No LLM API key found. LLM analysis will be disabled.")
            return None
    except Exception as e:
        st.error(f"Error loading LLM analyzer: {e}")
        return None

# Function to make predictions
def predict_url(url, model, llm_analyzer=None):
    """Make a prediction using both ML model (SVM) and LLM (if available)"""
    # First check if it's a trusted domain
    if is_trusted_domain(url):
        # For trusted domains, return legitimate with high confidence
        return 0, [0.99, 0.01], extract_features_from_urls([url]), None
    
    # Extract features for ML model
    features_df = extract_features_from_urls([url])
    
    # Handle feature name mismatch
    if hasattr(model, 'feature_names_in_'):
        # Check if we need to reorder columns
        required_features = model.feature_names_in_
        
        # Create a DataFrame with the required features, filled with zeros
        aligned_df = pd.DataFrame(0, index=[0], columns=required_features)
        
        # Fill in the values we have
        for col in features_df.columns:
            if col in required_features:
                aligned_df[col] = features_df[col].values
        
        features_df = aligned_df
    
    # Make SVM prediction
    if hasattr(model, 'named_steps'):
        ml_prediction = model.predict(features_df)[0]
        ml_probability = model.predict_proba(features_df)[0]
    else:
        ml_prediction = model.predict(features_df)[0]
        ml_probability = model.predict_proba(features_df)[0]
    
    # Apply custom rules
    ml_prediction, ml_probability = apply_custom_rules(url, ml_prediction, ml_probability, features_df)
    
    # If LLM analyzer is available, prioritize its analysis
    llm_analysis = None
    if llm_analyzer:
        try:
            # Get LLM prediction
            llm_prediction, llm_confidence = llm_analyzer.get_phishing_probability(url)
            llm_analysis = llm_analyzer.analyze_url(url)
            
            # Give more weight to LLM analysis for SVM model
            # SVM can be more sensitive to feature variations, so LLM can provide stability
            llm_weight = 0.7  # Higher weight for LLM
            ml_weight = 0.3   # Lower weight for SVM
            
            # Combine predictions with weights
            phishing_prob = (ml_probability[1] * ml_weight) + (llm_confidence * llm_weight if llm_prediction == 1 else (1 - llm_confidence) * llm_weight)
            
            final_prediction = 1 if phishing_prob > 0.5 else 0
            final_probability = [1 - phishing_prob, phishing_prob]
            
            # If LLM is very confident (>0.8), use its prediction
            if llm_confidence > 0.8:
                final_prediction = llm_prediction
                final_probability = [1 - llm_confidence, llm_confidence] if llm_prediction == 1 else [llm_confidence, 1 - llm_confidence]
                
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            final_prediction = ml_prediction
            final_probability = ml_probability
    else:
        final_prediction = ml_prediction
        final_probability = ml_probability
    
    return final_prediction, final_probability, features_df, llm_analysis

# Function to display feature importance
def display_feature_importance(features_df, model):
    try:
        # Check if it's an SVM model with a linear kernel
        if (hasattr(model, 'named_steps') and 'svm' in model.named_steps and 
            hasattr(model.named_steps['svm'], 'kernel') and 
            model.named_steps['svm'].kernel == 'linear'):
            
            # For linear SVM, use coefficients as feature importance
            feature_names = features_df.columns
            coefficients = model.named_steps['svm'].coef_[0]
            
            # Use absolute values for importance
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(coefficients)
            }).sort_values('importance', ascending=False)
            
            # Display top 10 features
            st.subheader("Top 10 Features for Phishing Detection")
            
            # Create a horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            top_n = min(10, len(feature_importance))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(top_n), ax=ax)
            ax.set_title('Feature Importance (SVM Coefficients)')
            st.pyplot(fig)
            
            # Display feature values for the current URL
            st.subheader("Feature Values for This URL")
            
            # Get top important features
            top_features = feature_importance.head(top_n)['feature'].tolist()
            
            # Display values for top features
            try:
                if all(feature in features_df.columns for feature in top_features):
                    top_features_df = features_df[top_features].T.reset_index()
                    top_features_df.columns = ['Feature', 'Value']
                    st.table(top_features_df)
                else:
                    st.write("Feature values not available for display")
            except Exception as e:
                st.warning(f"Could not display feature values: {e}")
                
        elif (hasattr(model, 'named_steps') and 'svm' in model.named_steps):
            # For non-linear SVM, we can't directly show feature importance
            st.info("Feature importance visualization is not available for non-linear SVM kernels.")
            
            # Instead, show the most important features based on domain knowledge
            st.subheader("Key URL Features for Phishing Detection")
            key_features = [
                "url_length", "dots_count", "suspicious_words_count", 
                "has_ip_address", "has_https", "domain_age_days",
                "is_common_domain", "has_suspicious_tld", "brand_in_subdomain"
            ]
            
            # Filter to only include features that exist in the dataframe
            available_features = [f for f in key_features if f in features_df.columns]
            
            if available_features:
                # Display values for these features
                key_features_df = features_df[available_features].T.reset_index()
                key_features_df.columns = ['Feature', 'Value']
                st.table(key_features_df)
            else:
                st.write("Feature values not available for display")
                
        else:
            # For other models, use the existing code
            # Get feature importance from model
            if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
                # For pipeline
                classifier = model.named_steps['classifier']
                if hasattr(model, 'named_steps') and 'feature_selection' in model.named_steps:
                    # If we used feature selection, get the selected features
                    feature_selector = model.named_steps['feature_selection']
                    if hasattr(feature_selector, 'get_support'):
                        feature_mask = feature_selector.get_support()
                        if hasattr(model, 'feature_names_in_'):
                            feature_names = np.array(model.feature_names_in_)[feature_mask]
                        else:
                            feature_names = np.array(features_df.columns)[feature_mask]
                        feature_importance = pd.DataFrame({
                            'feature': feature_names,
                            'importance': classifier.feature_importances_
                        }).sort_values('importance', ascending=False)
                    else:
                        feature_importance = pd.DataFrame({
                            'feature': features_df.columns,
                            'importance': classifier.feature_importances_
                        }).sort_values('importance', ascending=False)
                else:
                    feature_importance = pd.DataFrame({
                        'feature': features_df.columns,
                        'importance': classifier.feature_importances_
                    }).sort_values('importance', ascending=False)
            elif hasattr(model, 'feature_names_in_') and hasattr(model, 'feature_importances_'):
                # Make sure feature_names_in_ and feature_importances_ have the same length
                feature_names = model.feature_names_in_
                importances = model.feature_importances_
                
                if len(feature_names) == len(importances):
                    feature_importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                else:
                    # If lengths don't match, use indices as feature names
                    feature_importance = pd.DataFrame({
                        'feature': [f"Feature {i}" for i in range(len(importances))],
                        'importance': importances
                    }).sort_values('importance', ascending=False)
            else:
                # Fallback to using the features_df columns
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    if len(features_df.columns) == len(importances):
                        feature_importance = pd.DataFrame({
                            'feature': features_df.columns,
                            'importance': importances
                        }).sort_values('importance', ascending=False)
                    else:
                        # If lengths don't match, use indices as feature names
                        feature_importance = pd.DataFrame({
                            'feature': [f"Feature {i}" for i in range(len(importances))],
                            'importance': importances
                        }).sort_values('importance', ascending=False)
                else:
                    # If no feature importances available, create a dummy DataFrame
                    st.warning("Feature importance not available for this model")
                    return
            
            # Display top 10 features
            st.subheader("Top 10 Features for Phishing Detection")
            
            # Create a horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            top_n = min(10, len(feature_importance))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(top_n), ax=ax)
            ax.set_title('Feature Importance')
            st.pyplot(fig)
            
            # Display feature values for the current URL
            st.subheader("Feature Values for This URL")
            
            # Get top important features
            top_features = feature_importance.head(top_n)['feature'].tolist()
            
            # Display values for top features
            # Handle the case where features_df might have different column names
            try:
                if all(feature in features_df.columns for feature in top_features):
                    top_features_df = features_df[top_features].T.reset_index()
                    top_features_df.columns = ['Feature', 'Value']
                    st.table(top_features_df)
                else:
                    # Just show what we have
                    st.write("Feature values not available for display")
            except Exception as e:
                st.warning(f"Could not display feature values: {e}")
    
    except Exception as e:
        st.error(f"Error displaying feature importance: {e}")
        # Provide a fallback visualization
        st.write("Could not display feature importance due to an error.")

# Function to analyze URL and provide detailed explanation
def analyze_url(url):
    """Analyze URL and return detailed information"""
    analysis = {}
    
    # Parse URL
    parsed = urlparse(url)
    extracted = tldextract.extract(url)
    
    # Basic URL info
    analysis['protocol'] = parsed.scheme if parsed.scheme else "None"
    analysis['domain'] = extracted.domain
    analysis['tld'] = extracted.suffix
    analysis['subdomain'] = extracted.subdomain if extracted.subdomain else "None"
    
    # Security checks
    analysis['has_https'] = "Yes" if parsed.scheme == 'https' else "No"
    
    # Check for IP address
    ip_pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
    analysis['is_ip_address'] = "Yes" if ip_pattern.search(url) else "No"
    
    # Check for common legitimate domains
    common_domains = ['google', 'facebook', 'amazon', 'apple', 'microsoft', 'youtube', 
                     'twitter', 'instagram', 'linkedin', 'netflix', 'github', 'yahoo',
                     'wikipedia', 'reddit', 'ebay', 'paypal', 'spotify', 'twitch']
    
    analysis['is_common_domain'] = "Yes" if extracted.domain.lower() in common_domains else "No"
    
    # Check for suspicious TLDs
    suspicious_tlds = ['zip', 'cricket', 'link', 'work', 'party', 'gq', 'kim', 'country', 'science',
                       'tk', 'bid', 'ml', 'ga', 'cf', 'pw', 'top', 'xyz', 'date', 'faith', 'review']
    
    analysis['has_suspicious_tld'] = "Yes" if extracted.suffix in suspicious_tlds else "No"
    
    # Check for brand names in domain that aren't actually the brand's domain
    brand_names = ['paypal', 'apple', 'amazon', 'microsoft', 'facebook', 'google', 'twitter', 
                  'instagram', 'netflix', 'linkedin', 'ebay', 'spotify']
    
    domain = extracted.domain.lower()
    
    # Check if domain contains a brand name but isn't the actual brand domain
    for brand in brand_names:
        if brand in domain and domain != brand:
            analysis['brand_in_domain'] = f"Yes (contains '{brand}')"
            break
    else:
        analysis['brand_in_domain'] = "No"
    
    # Try to get domain age
    try:
        w = whois.whois(f"{extracted.domain}.{extracted.suffix}")
        if w.creation_date:
            if isinstance(w.creation_date, list):
                creation_date = w.creation_date[0]
            else:
                creation_date = w.creation_date
                
            domain_age = (datetime.now() - creation_date).days
            analysis['domain_age'] = f"{domain_age} days"
        else:
            analysis['domain_age'] = "Unknown"
    except:
        analysis['domain_age'] = "Unknown"
    
    return analysis

# Add this function to your app.py
def is_trusted_domain(url):
    """Check if the URL belongs to a trusted domain"""
    extracted = tldextract.extract(url)
    domain = extracted.domain.lower()
    suffix = extracted.suffix.lower()
    
    # List of trusted domains with their TLDs
    trusted_domains = {
        'google': ['com', 'co.uk', 'de', 'fr', 'ca', 'jp', 'au', 'co.in'],
        'facebook': ['com', 'net'],
        'amazon': ['com', 'co.uk', 'de', 'fr', 'ca', 'jp', 'in'],
        'microsoft': ['com', 'net', 'org'],
        'apple': ['com'],
        'youtube': ['com'],
        'twitter': ['com'],
        'instagram': ['com'],
        'linkedin': ['com'],
        'github': ['com', 'io'],
        'netflix': ['com'],
        'wikipedia': ['org', 'com'],
        'yahoo': ['com'],
        'ebay': ['com'],
        'paypal': ['com'],
        'reddit': ['com'],
        'gmail': ['com'],
        'outlook': ['com'],
        'dropbox': ['com'],
        'wordpress': ['com', 'org'],
        'shopify': ['com'],
        'adobe': ['com'],
        'spotify': ['com'],
        'twitch': ['tv'],
        'pinterest': ['com'],
        'tumblr': ['com'],
        'medium': ['com']
    }
    
    # Check if domain is in trusted list and has a valid TLD
    if domain in trusted_domains and suffix in trusted_domains.get(domain, []):
        return True
    
    return False

# Add this function to your app.py
def apply_custom_rules(url, prediction, probability, features_df):
    """Apply custom rules to override model predictions in certain cases"""
    extracted = tldextract.extract(url)
    domain = extracted.domain.lower()
    suffix = extracted.suffix.lower()
    
    # Rule 1: Always mark as legitimate for major trusted domains
    if is_trusted_domain(url):
        return 0, [0.99, 0.01]
    
    # Rule 2: Always mark as phishing for specific patterns
    suspicious_patterns = [
        r'paypal.*\.(?!com|net)$',  # paypal with unusual TLD
        r'apple.*\.(?!com)$',       # apple with unusual TLD
        r'secure.*\.tk$',           # secure with .tk TLD
        r'bank.*\.(?!com|gov|org)$' # bank with unusual TLD
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, f"{domain}.{suffix}"):
            return 1, [0.01, 0.99]
    
    # Rule 3: Check for IP addresses in URL (almost always phishing)
    if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url):
        return 1, [0.05, 0.95]
    
    # Rule 4: Very old domains are likely legitimate
    if 'domain_age_days' in features_df.columns:
        domain_age = features_df['domain_age_days'].values[0]
        if domain_age > 3650:  # > 10 years
            if prediction == 1 and probability[1] < 0.8:
                return 0, [0.8, 0.2]
    
    # If no rules matched, return the original prediction
    return prediction, probability

# Main function
def main():
    """Main function to run the Streamlit app"""
    # Load model and metrics
    model = load_model()
    metrics = load_metrics()
    llm_analyzer = load_llm_analyzer()
    
    # Set up the sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses a Support Vector Machine (SVM) model combined with "
        "Large Language Model analysis to detect phishing URLs with high accuracy."
    )
    
    # Add model information to sidebar
    st.sidebar.subheader("Model Information")
    st.sidebar.markdown("""
    **Machine Learning Model:** Support Vector Machine (SVM)
    
    **Key Features:**
    - High-dimensional feature analysis
    - Non-linear pattern recognition
    - Robust against overfitting
    """)
    
    # Display metrics if available
    if all(metrics.values()):
        st.sidebar.subheader("Model Performance")
        for key, value in metrics.items():
            st.sidebar.text(f"{key}: {value:.4f}")
    
    # LLM information
    if llm_analyzer:
        st.sidebar.subheader("LLM Integration")
        st.sidebar.markdown(f"""
        **Model:** {llm_analyzer.model}
        
        **Provider:** {llm_analyzer.model_provider.capitalize()}
        """)
    
    # Main content
    st.title("URL Phishing Detection System")
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
        <h3>Hybrid Phishing Detection</h3>
        <p>This tool combines a <b>Support Vector Machine (SVM)</b> classifier with 
        <b>Large Language Model</b> analysis to detect phishing URLs with high accuracy.</p>
        <p>Enter a URL below to analyze it for phishing characteristics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # URL input
    url_input = st.text_input("Enter a URL to check:", "https://www.example.com")
    
    # Check button
    if st.button("Check URL"):
        if model is None:
            st.error("Model not loaded. Please check if the model file exists.")
            return
        
        # Show spinner while processing
        with st.spinner("Analyzing URL..."):
            # Add a small delay to show the spinner (optional)
            time.sleep(1)
            
            # Make prediction
            prediction, probability, features_df, llm_analysis = predict_url(url_input, model, llm_analyzer)
            
            # Get detailed URL analysis
            url_analysis = analyze_url(url_input)
            
            # Display result
            col1, col2 = st.columns(2)
            
            with col1:
                # Display the final prediction prominently
                if prediction == 1:
                    st.error("⚠️ This URL is likely a PHISHING attempt!")
                    st.warning(f"Confidence: {probability[1]:.2%}")
                else:
                    st.success("✅ This URL appears to be LEGITIMATE.")
                    st.info(f"Confidence: {probability[0]:.2%}")
                
                # Display URL details
                st.subheader("URL Details")
                st.code(url_input)
                
                # Display URL analysis
                st.subheader("URL Analysis")
                analysis_df = pd.DataFrame({
                    'Property': url_analysis.keys(),
                    'Value': url_analysis.values()
                })
                st.table(analysis_df)
            
            # If LLM analysis is available, display it prominently
            if llm_analysis:
                st.subheader("🤖 AI Analysis")
                
                llm_col1, llm_col2 = st.columns(2)
                
                with llm_col1:
                    # Create a colored box based on the LLM assessment
                    if llm_analysis['is_likely_phishing']:
                        st.markdown("""
                        <div style="background-color: #ffcccc; padding: 15px; border-radius: 5px;">
                            <h3 style="color: #cc0000;">⚠️ AI Assessment: Likely Phishing</h3>
                            <p>Confidence: {:.0%}</p>
                        </div>
                        """.format(llm_analysis['confidence_score']), unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background-color: #ccffcc; padding: 15px; border-radius: 5px;">
                            <h3 style="color: #006600;">✅ AI Assessment: Likely Legitimate</h3>
                            <p>Confidence: {:.0%}</p>
                        </div>
                        """.format(llm_analysis['confidence_score']), unsafe_allow_html=True)
                    
                    # Display risk factors
                    if llm_analysis['risk_factors']:
                        st.write("**Risk Factors:**")
                        for factor in llm_analysis['risk_factors']:
                            st.write(f"- {factor}")
                
                with llm_col2:
                    # Display explanation and recommendation
                    st.write("**AI Explanation:**")
                    st.write(llm_analysis['explanation'])
                    
                    st.write("**AI Recommendation:**")
                    st.write(llm_analysis['recommendation'])
            
            # Display SVM model analysis
            with col2:
                # Display feature importance
                st.subheader("SVM Model Analysis")
                display_feature_importance(features_df, model)
                
                # Display explanation from ML model
                st.subheader("SVM-Based Explanation")
                if prediction == 1:
                    explanation = "The Support Vector Machine model identified this URL as likely phishing based on these characteristics:\n\n"
                    
                    # Add specific reasons based on features
                    if url_analysis['has_https'] == "No":
                        explanation += "- Uses insecure HTTP protocol instead of HTTPS\n"
                    if url_analysis['is_ip_address'] == "Yes":
                        explanation += "- Uses an IP address instead of a domain name\n"
                    if url_analysis['has_suspicious_tld'] == "Yes":
                        explanation += "- Uses a suspicious top-level domain\n"
                    if url_analysis['brand_in_domain'] != "No":
                        explanation += f"- {url_analysis['brand_in_domain']}, which may be an impersonation attempt\n"
                    if url_analysis['is_common_domain'] == "No":
                        explanation += "- Not a commonly recognized domain\n"
                    if url_analysis['domain_age'] == "Unknown" or "days" in url_analysis['domain_age'] and int(url_analysis['domain_age'].split()[0]) < 90:
                        explanation += "- Domain is new or age couldn't be verified\n"
                    
                    explanation += "\nBe cautious and avoid entering personal information on this website."
                    st.write(explanation)
                else:
                    explanation = "This URL appears to be legitimate based on its characteristics:\n\n"
                    
                    # Add specific reasons based on features
                    if url_analysis['has_https'] == "Yes":
                        explanation += "- Uses secure HTTPS protocol\n"
                    if url_analysis['is_ip_address'] == "No":
                        explanation += "- Uses a proper domain name instead of an IP address\n"
                    if url_analysis['has_suspicious_tld'] == "No":
                        explanation += "- Uses a standard top-level domain\n"
                    if url_analysis['brand_in_domain'] == "No":
                        explanation += "- No brand impersonation detected\n"
                    if url_analysis['is_common_domain'] == "Yes":
                        explanation += "- Is a commonly recognized domain\n"
                    if url_analysis['domain_age'] != "Unknown" and "days" in url_analysis['domain_age'] and int(url_analysis['domain_age'].split()[0]) > 90:
                        explanation += f"- Domain has been registered for {url_analysis['domain_age']}\n"
                    
                    explanation += "\nHowever, always exercise caution when sharing sensitive information online."
                    st.write(explanation)
            
            # Add this after displaying both analyses
            st.subheader("🔍 SVM + LLM Combined Analysis")

            # Create a progress bar showing the phishing probability
            st.write("**Phishing Probability:**")
            phishing_percentage = probability[1] * 100
            st.progress(phishing_percentage / 100)
            st.write(f"{phishing_percentage:.1f}%")

            # Show which factors contributed most to the decision
            st.write("**Key Factors in This Analysis:**")

            # Combine factors from both SVM and LLM
            key_factors = []

            # Add SVM factors
            if url_analysis['has_https'] == "No":
                key_factors.append("SVM: Uses insecure HTTP protocol")
            if url_analysis['is_ip_address'] == "Yes":
                key_factors.append("SVM: Uses an IP address instead of a domain name")
            if url_analysis['brand_in_domain'] != "No":
                key_factors.append(f"SVM: Contains brand name in domain ({url_analysis['brand_in_domain']})")

            # Add LLM factors if available
            if llm_analysis and llm_analysis['risk_factors']:
                # Add up to 3 risk factors from LLM
                for factor in llm_analysis['risk_factors'][:3]:
                    if factor not in key_factors:  # Avoid duplicates
                        key_factors.append(f"LLM: {factor}")

            # Display factors
            for factor in key_factors:
                st.write(f"- {factor}")

            # Final recommendation
            st.write("**Final Recommendation:**")
            if prediction == 1:
                st.markdown("""
                <div style="background-color: #ffcccc; padding: 15px; border-radius: 5px;">
                    <p>⚠️ <strong>Exercise extreme caution with this URL.</strong> It shows multiple characteristics 
                    of a phishing attempt. Avoid entering personal information or credentials.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background-color: #ccffcc; padding: 15px; border-radius: 5px;">
                    <p>✅ <strong>This URL appears to be legitimate.</strong> However, always be cautious 
                    when sharing sensitive information online.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Log prediction
        logging.info(f"URL: {url_input}, Prediction: {'Phishing' if prediction == 1 else 'Legitimate'}, Confidence: {max(probability):.2f}")
    
    # Example URLs
    st.subheader("Try these examples:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Legitimate URLs:")
        legitimate_examples = [
            "https://www.google.com",
            "https://facebook.com",
            "https://github.com",
            "https://www.amazon.com",
            "https://www.microsoft.com"
        ]
        
        for example in legitimate_examples:
            if st.button(example, key=f"leg_{example}"):
                st.session_state.url_input = example
                st.experimental_rerun()
    
    with col2:
        st.write("Suspicious URLs (for demonstration):")
        phishing_examples = [
            "http://paypal-secure.com.verify.account.login.php",
            "http://secure-banking.com/login/verify",
            "http://192.168.1.1/admin",
            "http://apple-icloud.signin.com/verify",
            "http://facebook-login.tk/auth"
        ]
        
        for example in phishing_examples:
            if st.button(example, key=f"phish_{example}"):
                st.session_state.url_input = example
                st.experimental_rerun()
    
    # Educational section
    st.subheader("How to Identify Phishing URLs")
    st.write("""
    Here are some tips to manually identify phishing URLs:
    
    1. **Check the URL carefully**: Phishing URLs often mimic legitimate websites with slight variations.
    2. **Look for HTTPS**: Legitimate websites typically use secure connections (https://).
    3. **Beware of IP addresses**: URLs with IP addresses instead of domain names are suspicious.
    4. **Check for misspellings**: Phishers often use domains with misspellings (e.g., "gooogle.com").
    5. **Be cautious of URL shorteners**: They can hide the actual destination.
    6. **Watch for excessive subdomains**: For example, "paypal.secure.com" is not the same as "paypal.com".
    7. **Check for unusual TLDs**: Be wary of unusual top-level domains like .tk, .xyz, etc.
    """)

if __name__ == "__main__":
    main() 