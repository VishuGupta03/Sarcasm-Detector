import csv
import random
import numpy as np
from textblob import TextBlob
from sklearn.naive_bayes import GaussianNB

def extract_features(row):
    """Extract features from a row of data"""
    try:
        features = []
        sentence = TextBlob(row['body'])
        features.append(sentence.sentiment.polarity)
        features.append(sentence.sentiment.subjectivity)
        score_2p, score_2s = calculate_ngram_scores(row['body'], 2)
        features.append(score_2p)
        features.append(score_2s)
        score_3p, score_3s = calculate_ngram_scores(row['body'], 3)
        features.append(score_3p)
        features.append(score_3s)
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return []

def calculate_ngram_scores(text, n):
    """Calculate sentiment scores for n-grams"""
    try:
        score_p, score_s = 0, 0
        word_count = 0
        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
            blob = TextBlob(ngram)
            try:
                score_p += blob.sentiment.polarity
                score_s += blob.sentiment.subjectivity
                word_count += 1
            except:
                word_count -= 1
        return score_p / (word_count + 0.0000001), score_s / (word_count + 0.0000001)
    except Exception as e:
        print(f"Error calculating n-gram scores: {e}")
        return 0, 0

def fetch_data(path):
    """Fetch data from a CSV file"""
    try:
        training_features, training_labels = [], []
        with open('reddit_training.csv', 'rt') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                features = extract_features(row)
                training_features.append(features)
                training_labels.append(row['sarcasm_tag'])
        return training_features, training_labels
    except Exception as e:
        print(f"Error fetching data: {e}")
        return [], []

def train_model(training_features, training_labels, model):
    """Train a Gaussian Naive Bayes model"""
    try:
        model.fit(training_features, training_labels)
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None

def test_model(model, testing_features):
    """Test a trained model"""
    try:
        return model.predict(testing_features)
    except Exception as e:
        print(f"Error testing model: {e}")
        return []

def print_test_passed(testing_labels, predicted_probability, testbody):
    """Print test cases with their scores"""
    try:
        for i in range(len(testing_labels)):
            if testing_labels[i] == 'yes':
                print(testbody[i], predicted_probability[i][1]*100)
    except Exception as e:
        print(f"Error printing test cases: {e}")

def main():
    print("Starting program...")
    model = GaussianNB()
    print("Fetching data...")
    training_features, training_labels = fetch_data('')
    print(f"Fetched {len(training_features)} training samples")
    print("Training model...")
    model = train_model(training_features, training_labels, model)
    print("Model trained")
    testing_features = []
    testbody = []
    with open('reddit_test.csv', 'rt') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            features = extract_features(row)
            testing_features.append(features)
            testbody.append(row['body'])
    print(f"Fetched {len(testing_features)} test samples")
    print("Making predictions...")
    testing_labels = test_model(model, testing_features)
    predicted_probability = model.predict_proba(testing_features)
    print("Printing test cases...")
    print_test_passed(testing_labels, predicted_probability, testbody)

if __name__ == "__main__":
    main()