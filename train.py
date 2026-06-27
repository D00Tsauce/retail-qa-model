import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

def load_and_prepare_data(file_path):
    """Loads the dataset and ensures columns exist."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
        
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['question', 'question_type'])
    
    X = df['question']      
    y = df['question_type'] 
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def main():
    data_path = os.path.join("data", "test.csv")
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data(data_path)
    
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    
    # UPGRADE: Added ngram_range=(1, 2) to capture word pairs, and sublinear_tf to smooth word counts
    print("Vectorizing text data with word pairs (bi-grams)...")
    vectorizer = TfidfVectorizer(
        max_features=10000, 
        stop_words='english', 
        ngram_range=(1, 2), 
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # UPGRADE: Using Support Vector Classifier with balanced class weights for text nuances
    print("Training Linear Support Vector Classifier...")
    model = LinearSVC(class_weight='balanced', random_state=42, C=1.0, max_iter=2000)
    model.fit(X_train_vec, y_train)
    
    # Evaluate performance
    y_pred = model.predict(X_test_vec)
    print("\n=== Model Performance Evaluation ===")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()