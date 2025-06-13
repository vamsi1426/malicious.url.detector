import pandas as pd
import numpy as np
import re
import urllib.parse
import tldextract
from urllib.parse import urlparse
import socket
import whois
import ssl
import requests
from datetime import datetime
import time

def extract_features_from_urls(urls):
    """
    Extract features from a list of URLs for phishing detection
    
    Args:
        urls (list): List of URL strings
        
    Returns:
        pandas.DataFrame: DataFrame containing extracted features
    """
    features = []
    
    for url in urls:
        # Initialize feature dictionary for this URL
        url_features = {}
        
        try:
            # Basic URL properties
            url_features.update(extract_basic_features(url))
            
            # Domain-based features
            url_features.update(extract_domain_features(url))
            
            # Address bar features
            url_features.update(extract_address_bar_features(url))
            
            # Security features
            url_features.update(extract_security_features(url))
            
            # Add advanced features
            url_features.update(extract_advanced_features(url))
            
            features.append(url_features)
        except Exception as e:
            print(f"Error extracting features from URL {url}: {e}")
            # Add a row with default values
            features.append(get_default_features())
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features)
    
    return features_df

def extract_basic_features(url):
    """Extract basic features from a URL"""
    features = {}
    
    # URL length
    features['url_length'] = len(url)
    
    # Number of dots in URL
    features['dots_count'] = url.count('.')
    
    # Number of slashes in URL
    features['slashes_count'] = url.count('/')
    
    # Number of question marks in URL
    features['question_mark_count'] = url.count('?')
    
    # Number of equal signs in URL
    features['equal_sign_count'] = url.count('=')
    
    # Number of digits in URL
    features['digits_count'] = sum(c.isdigit() for c in url)
    
    # Number of hyphens in URL
    features['hyphen_count'] = url.count('-')
    
    # Number of underscores in URL
    features['underscore_count'] = url.count('_')
    
    # Number of '@' symbols in URL
    features['at_symbol_count'] = url.count('@')
    
    # Number of ampersands in URL
    features['ampersand_count'] = url.count('&')
    
    # Number of percent signs in URL
    features['percent_count'] = url.count('%')
    
    # URL entropy (randomness measure)
    features['url_entropy'] = calculate_entropy(url)
    
    return features

def extract_domain_features(url):
    """Extract domain-based features from a URL"""
    features = {}
    
    # Parse the URL
    extracted = tldextract.extract(url)
    domain = extracted.domain
    suffix = extracted.suffix
    subdomain = extracted.subdomain
    
    # Domain length
    features['domain_length'] = len(domain) if domain else 0
    
    # Subdomain length
    features['subdomain_length'] = len(subdomain) if subdomain else 0
    
    # TLD length
    features['tld_length'] = len(suffix) if suffix else 0
    
    # Has subdomain
    features['has_subdomain'] = 1 if subdomain else 0
    
    # Number of subdomains
    if subdomain:
        features['subdomain_count'] = subdomain.count('.') + 1
    else:
        features['subdomain_count'] = 0
    
    # Domain contains digit
    features['domain_contains_digit'] = 1 if any(c.isdigit() for c in domain) else 0
    
    # Domain contains hyphen
    features['domain_contains_hyphen'] = 1 if '-' in domain else 0
    
    # Check for common legitimate domains
    common_domains = [
        'google', 'facebook', 'amazon', 'apple', 'microsoft', 'youtube', 
        'twitter', 'instagram', 'linkedin', 'netflix', 'github', 'yahoo',
        'wikipedia', 'reddit', 'ebay', 'paypal', 'spotify', 'twitch',
        'dropbox', 'adobe', 'cnn', 'bbc', 'nytimes', 'walmart', 'target',
        'gmail', 'outlook', 'hotmail', 'live', 'bing', 'pinterest', 'tumblr',
        'snapchat', 'whatsapp', 'telegram', 'tiktok', 'zoom', 'slack',
        'vimeo', 'quora', 'medium', 'wordpress', 'shopify', 'etsy'
    ]
    
    features['is_common_domain'] = 5 if domain.lower() in common_domains else 0
    
    # Check for domain age if possible
    try:
        w = whois.whois(f"{domain}.{suffix}")
        if w.creation_date:
            if isinstance(w.creation_date, list):
                creation_date = w.creation_date[0]
            else:
                creation_date = w.creation_date
                
            domain_age = (datetime.now() - creation_date).days
            features['domain_age_days'] = domain_age
            
            # Add a feature for old domains (likely legitimate)
            features['is_old_domain'] = 1 if domain_age > 365 else 0
        else:
            features['domain_age_days'] = -1
            features['is_old_domain'] = 0
    except:
        features['domain_age_days'] = -1
        features['is_old_domain'] = 0
    
    # If domain age check fails but it's a common domain, 
    # assume it's old (handles whois failures for legitimate sites)
    if features['is_common_domain'] > 0:
        features['is_old_domain'] = 1
    
    return features

def extract_address_bar_features(url):
    """Extract address bar features from a URL"""
    features = {}
    
    # Check for IP address in URL
    ip_pattern = re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
    features['has_ip_address'] = 1 if ip_pattern.search(url) else 0
    
    # Check for HTTPS
    features['has_https'] = 1 if url.startswith('https://') else 0
    
    # Check for URL shortening services
    shortening_services = ['bit.ly', 'goo.gl', 't.co', 'tinyurl', 'is.gd', 'cli.gs', 'ow.ly', 
                          'rebrand.ly', 'tiny.cc', 'shorte.st', 'go2l.ink', 'x.co', 'prettylinkpro.com',
                          'cutt.ly', 'tr.im', 'link.to']
    
    extracted = tldextract.extract(url)
    domain_with_suffix = f"{extracted.domain}.{extracted.suffix}"
    
    features['is_shortened'] = 1 if any(service in domain_with_suffix for service in shortening_services) else 0
    
    # Check for redirect in URL
    features['has_redirect'] = 1 if '//' in url.replace('://', '') else 0
    
    # Check for prefix/suffix in domain
    features['has_prefix_suffix'] = 1 if '-' in extracted.domain else 0
    
    # URL path length
    parsed = urlparse(url)
    features['path_length'] = len(parsed.path)
    
    # Number of path segments
    features['path_segment_count'] = len(parsed.path.split('/')) - 1
    
    # Query string length
    features['query_length'] = len(parsed.query)
    
    # Number of parameters in query
    features['parameter_count'] = len(parsed.query.split('&')) if parsed.query else 0
    
    return features

def extract_security_features(url):
    """Extract security-related features from a URL"""
    features = {}
    
    # Check for suspicious words in URL
    suspicious_words = ['secure', 'account', 'webscr', 'login', 'ebayisapi', 'signin', 'banking', 
                        'confirm', 'update', 'verify', 'password', 'pay', 'wallet', 'alert', 
                        'purchase', 'transaction', 'recover', 'unlock', 'authorize', 'authenticate']
    
    url_lower = url.lower()
    features['suspicious_words_count'] = 3 * sum(word in url_lower for word in suspicious_words)
    
    # Check for suspicious TLDs
    suspicious_tlds = ['zip', 'cricket', 'link', 'work', 'party', 'gq', 'kim', 'country', 'science',
                       'tk', 'bid', 'ml', 'ga', 'cf', 'pw', 'top', 'xyz', 'date', 'faith', 'review']
    
    extracted = tldextract.extract(url)
    features['has_suspicious_tld'] = 1 if extracted.suffix in suspicious_tlds else 0
    
    # Check for brand names in domain that aren't actually the brand's domain
    brand_names = ['paypal', 'apple', 'amazon', 'microsoft', 'facebook', 'google', 'twitter', 
                  'instagram', 'netflix', 'linkedin', 'ebay', 'spotify']
    
    domain = extracted.domain.lower()
    
    # Check if domain contains a brand name but isn't the actual brand domain
    for brand in brand_names:
        if brand in domain and domain != brand:
            features['brand_in_subdomain'] = 1
            break
    else:
        features['brand_in_subdomain'] = 0
    
    # Check for SSL certificate
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc
        if not hostname:
            # If netloc is empty, use the path (common for incomplete URLs)
            hostname = parsed.path.split('/')[0]
        
        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=3) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                # Certificate exists
                features['has_ssl'] = 1
                
                # Check certificate expiration
                expire_date = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                days_to_expire = (expire_date - datetime.now()).days
                features['ssl_days_to_expire'] = days_to_expire
    except:
        features['has_ssl'] = 0
        features['ssl_days_to_expire'] = -1
    
    # In extract_security_features, add more suspicious patterns
    suspicious_patterns = [
        r'paypal.*\.(?!com|net)', # paypal followed by any TLD except .com or .net
        r'apple.*\.(?!com|ca|co\.uk)', # apple with unusual TLD
        r'facebook.*\.(?!com|net)', # facebook with unusual TLD
        r'microsoft.*\.(?!com|net|org)', # microsoft with unusual TLD
        r'verify.*login', # verify and login in the same URL
        r'secure.*\.(?!gov|edu|mil)' # secure with non-institutional TLD
    ]

    for pattern in suspicious_patterns:
        if re.search(pattern, url.lower()):
            features['has_suspicious_pattern'] = 1
            break
    else:
        features['has_suspicious_pattern'] = 0
    
    return features

def extract_advanced_features(url):
    """Extract advanced features that might require external requests"""
    features = {}
    
    try:
        # Try to make a request to the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=5, allow_redirects=True)
        
        # Check if there was a redirect
        features['was_redirected'] = 1 if len(response.history) > 0 else 0
        
        # Final URL after potential redirects
        final_url = response.url
        features['redirect_count'] = len(response.history)
        
        # Check if the domain changed after redirect
        original_domain = tldextract.extract(url).registered_domain
        final_domain = tldextract.extract(final_url).registered_domain
        features['domain_changed'] = 1 if original_domain != final_domain else 0
        
        # Check for iframe usage (often used in phishing)
        features['has_iframe'] = 1 if '<iframe' in response.text.lower() else 0
        
        # Check for form submission (common in phishing)
        features['has_form'] = 1 if '<form' in response.text.lower() else 0
        
        # Check for password input fields
        features['has_password_field'] = 1 if 'type="password"' in response.text.lower() else 0
        
        # Check for JavaScript redirects
        features['has_js_redirect'] = 1 if 'window.location' in response.text.lower() else 0
        
    except:
        # If request fails, set default values
        features['was_redirected'] = 0
        features['redirect_count'] = 0
        features['domain_changed'] = 0
        features['has_iframe'] = 0
        features['has_form'] = 0
        features['has_password_field'] = 0
        features['has_js_redirect'] = 0
    
    return features

def calculate_entropy(string):
    """Calculate the entropy of a string"""
    # Get probability of chars
    prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
    # Calculate entropy
    entropy = -sum([p * np.log2(p) for p in prob])
    return entropy

def get_default_features():
    """Return a dictionary with default feature values"""
    return {
        # Basic features
        'url_length': 0,
        'dots_count': 0,
        'slashes_count': 0,
        'question_mark_count': 0,
        'equal_sign_count': 0,
        'digits_count': 0,
        'hyphen_count': 0,
        'underscore_count': 0,
        'at_symbol_count': 0,
        'ampersand_count': 0,
        'percent_count': 0,
        'url_entropy': 0,
        
        # Domain features
        'domain_length': 0,
        'subdomain_length': 0,
        'tld_length': 0,
        'has_subdomain': 0,
        'subdomain_count': 0,
        'domain_contains_digit': 0,
        'domain_contains_hyphen': 0,
        'is_common_domain': 0,
        'domain_age_days': -1,
        'is_old_domain': 0,
        
        # Address bar features
        'has_ip_address': 0,
        'has_https': 0,
        'is_shortened': 0,
        'has_redirect': 0,
        'has_prefix_suffix': 0,
        'path_length': 0,
        'path_segment_count': 0,
        'query_length': 0,
        'parameter_count': 0,
        
        # Security features
        'suspicious_words_count': 0,
        'has_suspicious_tld': 0,
        'brand_in_subdomain': 0,
        'has_ssl': 0,
        'ssl_days_to_expire': -1,
        'has_suspicious_pattern': 0,
        
        # Advanced features
        'was_redirected': 0,
        'redirect_count': 0,
        'domain_changed': 0,
        'has_iframe': 0,
        'has_form': 0,
        'has_password_field': 0,
        'has_js_redirect': 0
    }

# Test the feature extraction if run directly
if __name__ == "__main__":
    test_urls = [
        "https://www.google.com",
        "http://malicious-phishing-site.com/login/secure/paypal.com",
        "https://bit.ly/2Vxp5gK",
        "http://192.168.1.1/admin",
        "https://www.paypal-secure.com/account/login.php"
    ]
    
    features_df = extract_features_from_urls(test_urls)
    print(features_df) 