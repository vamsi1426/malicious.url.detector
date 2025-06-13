# URL Phishing Detection System

A machine learning and LLM-powered system to detect phishing URLs with high accuracy. This system combines traditional ML techniques with advanced language model analysis to provide comprehensive phishing detection.

## Features

- **Hybrid Detection**: Combines machine learning models with Large Language Model (LLM) analysis
- **Detailed Analysis**: Provides comprehensive breakdown of URL characteristics
- **Visual Explanations**: Shows feature importance and risk factors
- **User-Friendly Interface**: Easy-to-use Streamlit web application
- **Educational Content**: Helps users understand phishing techniques

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/url-phishing-detection.git
   cd url-phishing-detection
   ```

2. Create and activate a virtual environment (recommended):
   ```
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up API keys:
   - Create a `.env` file in the project root directory
   - Add your OpenRouter API key (or other LLM provider keys):
     ```
     OPENROUTER_API_KEY=your-api-key-here
     SITE_URL=http://localhost:8501
     SITE_NAME=URL Phishing Detector
     ```

## Usage

### Training the Model

Before running the application, you need to train the machine learning model: 

## Machine Learning Model

This system uses a Support Vector Machine (SVM) classifier to detect phishing URLs. SVM was chosen for its:

- Effectiveness in high-dimensional spaces
- Strong performance on text and URL classification tasks
- Ability to handle non-linear decision boundaries with kernel functions
- Robustness against overfitting

The model is trained using a pipeline that includes:
1. Standard scaling of features (required for SVM)
2. SVM classification with optimized hyperparameters
3. SMOTE oversampling to handle class imbalance

### Model Training

The SVM model is trained using grid search to find optimal hyperparameters:
- Kernel: RBF or Linear
- C parameter: Controls regularization
- Gamma parameter: Controls kernel coefficient

### Feature Importance

For linear SVM kernels, feature importance is derived from the absolute values of the coefficients. For non-linear kernels (RBF), direct feature importance is not available, but the system uses domain knowledge to highlight key features.

## Hybrid Detection System

This system combines two powerful approaches for phishing detection:

### 1. Support Vector Machine (SVM) Model

The SVM classifier analyzes URL features such as:
- Domain characteristics (age, registration details)
- URL structure (length, special characters)
- Security indicators (HTTPS, certificates)
- Lexical features (suspicious words, brand names)

### 2. Large Language Model (LLM) Analysis

The system integrates with state-of-the-art language models through OpenRouter:
- Meta's Llama 3.3 70B model provides semantic understanding of URLs
- Contextual analysis of brand impersonation attempts
- Recognition of emerging phishing patterns
- Human-readable explanations of risk factors

### Combined Approach

The final verdict is determined by combining both analyses:
- SVM provides statistical pattern recognition
- LLM provides contextual understanding and reasoning
- Weighted combination gives more accurate results than either approach alone
- Detailed explanations help users understand the risk assessment

This hybrid approach is particularly effective because:
1. SVM excels at recognizing known patterns in URL features
2. LLM can identify novel phishing techniques and provide context
3. Together they provide both high accuracy and explainability