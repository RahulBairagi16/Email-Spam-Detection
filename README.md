# Email Spam Filtering Project

## Overview
This project implements a machine learning model to classify emails as spam or not spam, using Natural Language Processing (NLP) techniques. The model is designed to improve email security by effectively filtering out unwanted spam emails. By leveraging various text processing techniques and machine learning algorithms, the project provides an end-to-end solution for spam detection.

## Key Features
- **Natural Language Processing (NLP)**: Tokenization, text normalization, TF-IDF, word embeddings.
- **Machine Learning Models**: Naive Bayes, Random Forest.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score.
- **Python Libraries**: NLTK, Scikit-learn, Pandas, NumPy.

## Objective
The primary goal of this project is to design and implement a machine learning model that can classify emails as either spam or not spam with high accuracy. The model aims to minimize the false positive rate while maximizing precision and recall, ensuring that genuine emails are not mistakenly classified as spam.

## Dataset
The dataset used in this project consists of labeled emails that are categorized as either spam or not spam. Each email is treated as a document, and its content is processed for feature extraction using NLP techniques.

## Methodology

### 1. **Data Preprocessing**
- **Handling Missing Values**: Ensured all missing or corrupted data points were appropriately handled.
- **Tokenization**: Split email content into individual tokens (words or phrases) for further analysis.
- **Text Normalization**: Converted all text to lowercase, removed special characters, and eliminated stop words.
- **Feature Extraction**:
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**: Used to convert email content into numerical feature vectors.
  - **Word Embeddings**: Techniques like Word2Vec were explored to represent words in a continuous vector space, capturing semantic relationships.

### 2. **Model Development**
- **Naive Bayes Classifier**: Implemented this model as it works well with text data and is often effective for spam classification tasks.
- **Random Forest Classifier**: This model was used to compare results and provide robust classification by combining multiple decision trees.
- **Hyperparameter Tuning**: Applied techniques like Grid Search to fine-tune the models and improve performance.

### 3. **Model Evaluation**
- **Accuracy**: Evaluated the overall correctness of the model’s predictions.
- **Precision**: Measured how many emails predicted as spam were actually spam (important for minimizing false positives).
- **Recall**: Evaluated how well the model captured actual spam emails.
- **F1-Score**: Used to balance precision and recall for a more comprehensive evaluation.
- **Confusion Matrix**: Generated to analyze the classification performance in detail.

### 4. **Optimization**
- Improved feature selection using advanced NLP techniques like word embeddings and TF-IDF transformations.
- Optimized model performance by balancing bias and variance through hyperparameter tuning and model selection.

## Results
- The final **Naive Bayes classifier** achieved an accuracy of **97%** and a precision of **99%**.  
- This significantly reduced the number of spam emails and provided better email filtering with a very low false positive rate.
- The **Random Forest classifier** provided competitive results but was computationally heavier compared to Naive Bayes, making the latter a more efficient choice for this use case.

## Conclusion
This project successfully demonstrates how machine learning and NLP techniques can be used to classify emails as spam or not spam. The implemented model achieves high accuracy and precision, improving the user experience by reducing spam in email inboxes. The use of advanced text processing and machine learning algorithms allows the system to continually improve as more data becomes available.

## Future Improvements
- **Deep Learning Models**: Experiment with deep learning techniques such as LSTM or BERT to further improve accuracy and scalability.
- **Real-Time Classification**: Integrate the model into a real-time email system for live spam filtering.
- **Handling Imbalanced Data**: Explore techniques such as SMOTE or undersampling to handle imbalanced datasets more effectively.

## Installation

To run this project locally, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/rahulbairagi/email-spam-filtering.git
