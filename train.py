from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

def load_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Change encoding based on your file
    return df

def train_model(df):
    X = df['Text']
    y = df['Sentiment']  # Assuming the sentiment column is named 'Sentiment'

    # Fit the LabelEncoder on all the possible sentiments
    label_encoder = LabelEncoder()
    label_encoder.fit(y)

    # Encode the labels
    y_encoded = label_encoder.transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Vectorize the input text
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train the model
    model = LogisticRegression(max_iter=1000)  # Increased max_iter to avoid convergence warning
    model.fit(X_train_vec, y_train)

    # Calculate accuracy on the test set
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Model trained with accuracy: {accuracy}")
    print(f"Confusion Matrix:\n {cm}")

    # Save the model, label encoder, and vectorizer
    joblib.dump(model, 'sentiment_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    # Return model and test data for evaluation
    return model, label_encoder, X_test_vec, y_test

# Load and clean your data here
file_path = 'D:/Projects/SentimentAnalysis/Data/train.csv'  # Update with your actual file path
df_cleaned = load_data(file_path)

# Train the model and get test data
model, label_encoder, X_test_vec, y_test = train_model(df_cleaned)
