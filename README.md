# 📧 SMS Phishing Detection - End-to-End Report

Project Live Link: https://sms-phishing-data-sec2024.streamlit.app/

## Overview

The goal of this project is to build a **Phishing Message Detection System** that classifies messages as either **Phishing** or **Legitimate** based on their content. The system uses a machine learning model to predict the nature of a message after cleaning and preprocessing the text.

The project involves the following key steps:
1. Data Preprocessing
2. Feature Extraction using TF-IDF
3. Model Training using Random Forest Classifier
4. Model Evaluation
5. Model Deployment using Streamlit

This report covers the methodology, implementation, results, and deployment of the SMS Phishing Detection model.

---

## 1. Data Collection

The dataset used for this project is the **SMS Spam Collection Dataset**. It contains messages labeled as **spam** (phishing) and **ham** (legitimate). The publicly available dataset can be downloaded from sources like [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

The dataset consists of:
- **v1**: Label (`Legitimate` or `Phishing`)
- **v2**: Message content

### Example Entries:
| v1           | v2                                                                                      |
|--------------|-----------------------------------------------------------------------------------------|
| Legitimate   | "Hey, how about a game of golf tomorrow?"                                               |
| Phishing     | "Congratulations! You've won a $1000 gift card. Click here to claim your prize!"        |

The dataset contains 5,572 messages, with an imbalanced distribution of 747 Phishing messages and 4,825 Legitimate messages.

---

## 2. Data Preprocessing

Before feeding the data into a machine learning model, several preprocessing steps were performed:

### 2.1. Cleaning Text Data
We cleaned the message text to standardize it and remove irrelevant information:
- Convert text to lowercase.
- Remove digits.
- Remove URLs (e.g., `http://`, `www`).
- Remove punctuation marks.
- Remove extra whitespaces.

```python
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    return text
```
### 2.2. Feature Engineering
We used TF-IDF (Term Frequency-Inverse Document Frequency) to transform the cleaned text data into numerical features. This method captures the importance of a word about the entire dataset.

N-grams: We used bi-grams (pairs of consecutive words) to capture more context.
Stopwords Removal: Common words like "the", "a", etc., were removed.

```python
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
```

## 3. Model Training
We trained a Random Forest Classifier for this classification problem. Random Forest is a versatile and powerful model that uses multiple decision trees to make predictions.

### 3.1. Data Splitting
The dataset was split into training and testing sets (80% training, 20% testing) to evaluate the model's performance.

```python
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
```

### 3.2. Handling Class Imbalance with SMOTE
Since the dataset is imbalanced, we used SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic examples of the minority class (spam). This helped improve the model’s performance on the minority class.

```python
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

### 3.3. Model Training
We trained the Random Forest Classifier with 100 trees and a maximum depth of 10 for better generalization.

```python
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)
```

## 4. Model Evaluation
After training the model, we evaluated its performance on the test set. The evaluation metrics include accuracy, precision, recall, and the confusion matrix.

### 4.1. Performance Metrics
- Accuracy: 90.88%
- Precision (Spam): 1.00
- Recall (Spam): 0.82
- F1-score: 0.90
- Classification Report:

                  precision    recall  f1-score   support
               0       0.85      1.00      0.92       985
               1       1.00      0.82      0.90       945
    accuracy                               0.91      1930
    macro avg          0.92      0.91      0.91      1930
    weighted avg       0.92      0.91      0.91      1930

- Confusion Matrix:

[[982   3]

 [173 772]]
 
The confusion matrix indicates that the model performs well in identifying legitimate messages (True Negatives), but there are some misclassifications for spam messages (False Negatives).

## 5. Model Deployment
The model was deployed as a web application using Streamlit, allowing users to input a message and get a prediction on whether it's Phishing or Legitimate.

### 5.1. Streamlit Interface
The application provides the following features:

Message Input: Users can type a message into a text box.
Prediction Button: Once a message is entered, users can click the "Predict" button to classify the message.

```python
user_input = st.text_area("Type your message here:", height=150)
```

### 5.2. Result Display
The prediction is displayed with either a warning (for phishing) or a success message (for legitimate).

```python
if prediction == 1:
    st.error("⚠️ **This message appears to be Phishing.** Avoid clicking on links or sharing sensitive information.")
else:
    st.success("✅ **This message appears to be Legitimate.** No harmful indicators were detected.")
```

### 5.3. Streamlit Setup
To set up Streamlit, we use the following code:

```python
st.set_page_config(page_title="Message Phishing Detector", page_icon="📧")
st.title("📧 Message Phishing Detector")
st.markdown("This application helps you identify whether a message is **Phishing** or **Legitimate**.")
```

The app can be accessed via the following link: SMS Phishing Detection App.

## 6. Conclusion
The SMS Phishing Detection model performs well with an accuracy of over 90%. The combination of TF-IDF for feature extraction, SMOTE for handling class imbalance, and Random Forest for classification resulted in a robust model capable of detecting phishing messages with high precision.

Future improvements could include:

- Exploring deep learning models such as LSTM or BERT for better context understanding.
- Expanding the dataset for better generalization across different message types.
- Implementing real-time phishing detection in messaging applications.

## 7. Files Included
- data_processing.py: Contains code for data loading, cleaning, and preprocessing.
- main.py: Implements the Streamlit application and user interface.
- spam_model.pkl: The trained model used for predictions.
- vectorizer.pkl: The vectorizer used to transform text into numerical features.
- spam.csv: The dataset containing labeled SMS messages.

This Markdown report covers all aspects of the project, including the data collection, preprocessing, model training, evaluation, deployment, and conclusions.
