import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
import flask
from flask import Flask, request, render_template, jsonify
import logging
import datetime
# Initialize Flask app
app = Flask(__name__)
# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
# Logging configuration
logging.basicConfig(
    filename='spam_detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
class EmailSpamDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.models = {
            'naive_bayes': MultinomialNB(),
            'svm': SVC(kernel='linear', probability=True),
            'random_forest': RandomForestClassifier(n_estimators=100)
        }
        self.best_model = None
        self.best_accuracy = 0 
    def preprocess_text(self, text):
        """Clean and preprocess email text"""
        # Convert to lowercase
        text = text.lower()
         # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespaces
        text = ' '.join(text.split())
         # Tokenization and stemming
        words = text.split()
        words = [stemmer.stem(word) for word in words if word not in stop_words]
        return ' '.join(words)
    def load_data(self, filepath):
        """Load and preprocess email dataset"""
        try:
            df = pd.read_csv(filepath, encoding='latin-1')
            df = df[['v1', 'v2']]  # spam/ham labels and email text
            df.columns = ['label', 'email']
                        # Preprocess email text
            df['processed_email'] = df['email'].apply(self.preprocess_text)
            return df
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            return None
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple ML models and select the best performer"""
        results = {}
         for name, model in self.models.items():
            try:
                # Create pipeline
                pipeline = Pipeline([
                    ('tfidf', self.vectorizer),
                    ('classifier', model)
                ])
                 # Train model
                pipeline.fit(X_train, y_train)
          # Make predictions
                y_pred = pipeline.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                results[name] = {
                    'model': pipeline,
                    'accuracy': accuracy,
                    'predictions': y_pred
                }
                # Track best model
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_model = pipeline
                
                logging.info(f"{name} accuracy: {accuracy:.4f}")
                
            except Exception as e:
                logging.error(f"Error training {name}: {str(e)}")
        
        return results
    
    def predict_email(self, email_text):
        """Predict if single email is spam or ham"""
        if self.best_model is None:
            return None, None
        
        processed_text = self.preprocess_text(email_text)
        prediction = self.best_model.predict([processed_text])[0]
        probability = self.best_model.predict_proba([processed_text]).max()
        
        return prediction, probability
    
    def save_model(self, filepath):
        """Save trained model to file"""
        if self.best_model:
            with open(filepath, 'wb') as f:
                pickle.dump(self.best_model, f)
            logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load pre-trained model from file"""
        try:
            with open(filepath, 'rb') as f:
                self.best_model = pickle.load(f)
            logging.info(f"Model loaded from {filepath}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")

# Initialize detector
detector = EmailSpamDetector()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        email_text = request.form['email_text']
        prediction, confidence = detector.predict_email(email_text)
        
        result = {
            'prediction': prediction,
            'confidence': f"{confidence:.2%}",
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Log prediction
        logging.info(f"Prediction made: {result}")
        
        return jsonify(result)
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        
        predictions = []
        for email in df['email_text']:
            pred, conf = detector.predict_email(email)
            predictions.append({'prediction': pred, 'confidence': conf})
        
        return jsonify({'results': predictions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load and train model if not exists
    try:
        detector.load_model('spam_model.pkl')
    except:
        # Train new model
        data = detector.load_data('spam_dataset.csv')
        if data is not None:
            X = data['processed_email']
            y = data['label']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            detector.train_models(X_train, y_train, X_test, y_test)
            detector.save_model('spam_model.pkl')
    
    app.run(debug=True, host='0.0.0.0', port=5000)
