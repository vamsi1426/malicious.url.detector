import requests
import json
import os
import time
from typing import Dict, Any, List, Tuple

class LLMAnalyzer:
    """Class to handle LLM-based URL analysis"""
    
    def __init__(self, model_provider="openai", model_name=None):
        """Initialize the LLM analyzer with the specified provider"""
        self.model_provider = model_provider
        
        # Load API keys from environment variables
        if model_provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.api_url = "https://api.openai.com/v1/chat/completions"
            self.model = model_name or "gpt-3.5-turbo"
        elif model_provider == "meta":
            self.api_key = os.getenv("META_API_KEY")
            self.api_url = "https://llama-api.meta.com/v1/completions"
            self.model = model_name or "llama-3-8b"
        elif model_provider == "anthropic":
            self.api_key = os.getenv("ANTHROPIC_API_KEY")
            self.api_url = "https://api.anthropic.com/v1/messages"
            self.model = model_name or "claude-3-sonnet-20240229"
        elif model_provider == "openrouter":
            self.api_key = os.getenv("OPENROUTER_API_KEY")
            self.api_url = "https://openrouter.ai/api/v1/chat/completions"
            self.model = model_name or "meta-llama/llama-3.3-70b-instruct:free"
            self.site_url = os.getenv("SITE_URL", "http://localhost:8501")  # Default to Streamlit local URL
            self.site_name = os.getenv("SITE_NAME", "URL Phishing Detector")
        else:
            raise ValueError(f"Unsupported model provider: {model_provider}")
    
    def analyze_url(self, url: str) -> Dict[str, Any]:
        """
        Analyze a URL using the LLM to detect phishing characteristics
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Create prompt for the LLM
        prompt = self._create_prompt(url)
        
        # Get response from LLM
        llm_response = self._query_llm(prompt)
        
        # Parse the response
        analysis_result = self._parse_response(llm_response, url)
        
        return analysis_result
    
    def _create_prompt(self, url: str) -> Any:
        """Create a prompt for the LLM based on the URL"""
        system_content = """You are a cybersecurity expert specializing in phishing URL detection. 
        Analyze the provided URL for phishing characteristics. Consider:
        1. Domain name (suspicious misspellings, subdomains, TLDs)
        2. URL structure (suspicious paths, parameters)
        3. Brand impersonation attempts
        4. Use of URL shorteners or redirects
        5. Use of IP addresses instead of domain names
        6. Presence of suspicious keywords
        
        Provide your analysis in JSON format with the following fields:
        - is_likely_phishing: boolean
        - confidence_score: float (0.0 to 1.0)
        - risk_factors: array of strings
        - explanation: string
        - recommendation: string
        """
        
        user_content = f"Analyze this URL for phishing: {url}"
        
        if self.model_provider in ["openai", "openrouter"]:
            return [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
        elif self.model_provider in ["meta", "anthropic"]:
            return f"{system_content}\n\n{user_content}"
    
    def _query_llm(self, prompt: Any) -> str:
        """Send a query to the LLM API and get the response"""
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.model_provider == "openai":
            headers["Authorization"] = f"Bearer {self.api_key}"
            data = {
                "model": self.model,
                "messages": prompt,
                "temperature": 0.1,
                "max_tokens": 1000
            }
        elif self.model_provider == "meta":
            headers["Authorization"] = f"Bearer {self.api_key}"
            data = {
                "model": self.model,
                "prompt": prompt,
                "temperature": 0.1,
                "max_tokens": 1000
            }
        elif self.model_provider == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 1000
            }
        elif self.model_provider == "openrouter":
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["HTTP-Referer"] = self.site_url
            headers["X-Title"] = self.site_name
            data = {
                "model": self.model,
                "messages": prompt,
                "temperature": 0.1,
                "max_tokens": 1000
            }
        
        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            
            if self.model_provider in ["openai", "openrouter"]:
                return response.json()["choices"][0]["message"]["content"]
            elif self.model_provider == "meta":
                return response.json()["choices"][0]["text"]
            elif self.model_provider == "anthropic":
                return response.json()["content"][0]["text"]
            
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return "{}"
    
    def _parse_response(self, response: str, url: str) -> Dict[str, Any]:
        """Parse the LLM response into a structured format"""
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                result = json.loads(json_str)
            else:
                # Fallback if no JSON is found
                result = {
                    "is_likely_phishing": False,
                    "confidence_score": 0.5,
                    "risk_factors": [],
                    "explanation": "Could not parse LLM response",
                    "recommendation": "Please try again or use traditional analysis"
                }
            
            # Add the URL to the result
            result["url"] = url
            
            return result
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return {
                "url": url,
                "is_likely_phishing": False,
                "confidence_score": 0.5,
                "risk_factors": [],
                "explanation": f"Error parsing LLM response: {str(e)}",
                "recommendation": "Please try again or use traditional analysis"
            }
    
    def get_phishing_probability(self, url: str) -> Tuple[int, float]:
        """
        Get a simple phishing prediction and probability from the LLM
        
        Args:
            url: The URL to analyze
            
        Returns:
            Tuple of (prediction, probability) where:
            - prediction: 1 for phishing, 0 for legitimate
            - probability: Confidence score between 0 and 1
        """
        analysis = self.analyze_url(url)
        
        prediction = 1 if analysis.get("is_likely_phishing", False) else 0
        probability = analysis.get("confidence_score", 0.5)
        
        return prediction, probability


# Example usage
if __name__ == "__main__":
    # Set your API key as an environment variable before running
    # os.environ["OPENAI_API_KEY"] = "your-api-key"
    
    analyzer = LLMAnalyzer(model_provider="openai")
    
    test_urls = [
        "https://www.google.com",
        "http://paypal-secure.com.verify.account.login.php",
        "https://facebook.com",
        "http://192.168.1.1/admin"
    ]
    
    for url in test_urls:
        print(f"\nAnalyzing: {url}")
        analysis = analyzer.analyze_url(url)
        print(json.dumps(analysis, indent=2))
        
        prediction, probability = analyzer.get_phishing_probability(url)
        print(f"Prediction: {'Phishing' if prediction == 1 else 'Legitimate'}")
        print(f"Confidence: {probability:.2f}") 