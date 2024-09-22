import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load cleaned data and the model
def load_cleaned_data(file_path):
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    return df

def evaluate_model(df, model, label_encoder, vectorizer):
    X_test = df['Text']
    true_labels = df['Sentiment']

    # Vectorize the input text using the loaded vectorizer
    X_test_vec = vectorizer.transform(X_test)

    # Predict sentiment using the loaded model
    predicted_labels_encoded = model.predict(X_test_vec)

    # Encode true labels using the label encoder
    true_labels_encoded = label_encoder.transform(true_labels)

    # Accuracy
    accuracy = accuracy_score(true_labels_encoded, predicted_labels_encoded)
    print(f'Test Accuracy: {accuracy:.2f}')

    # Confusion Matrix
    cm = confusion_matrix(true_labels_encoded, predicted_labels_encoded)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

    # Classification Report
    report = classification_report(true_labels_encoded, predicted_labels_encoded, target_names=label_encoder.classes_)
    print(report)

if __name__ == '__main__':
    # Path to cleaned test data
    cleaned_data_path = 'D:/Projects/SentimentAnalysis/Data/test.csv'
    df_cleaned = load_cleaned_data(cleaned_data_path)

    # Load saved model, label encoder, and vectorizer
    model = joblib.load('sentiment_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    # Evaluate model on test data
    evaluate_model(df_cleaned, model, label_encoder, vectorizer)
