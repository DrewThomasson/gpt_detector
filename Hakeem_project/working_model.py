import os
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Download the required NLTK packages
nltk.download('punkt')

# Function to extract features from a paragraph
def extract_features(paragraph):
    sentences = sent_tokenize(paragraph)
    words = word_tokenize(paragraph)
    features = {
        "num_sentences": len(sentences),
        "num_words": len(words),
        "presence_parentheses": 1 if "(" in paragraph or ")" in paragraph else 0,
        "presence_dash": 1 if "-" in paragraph else 0,
        "presence_semicolon_colon": 1 if ";" in paragraph or ":" in paragraph else 0,
        "presence_question_mark": 1 if "?" in paragraph else 0,
        "presence_apostrophe": 1 if "'" in paragraph else 0,
        "std_sentence_length": np.std([len(sent) for sent in sentences]),
        "mean_diff_sentence_length": np.mean([abs(len(sentences[i]) - len(sentences[i-1])) 
                                              for i in range(1, len(sentences))]) if len(sentences) > 1 else 0,
        "presence_short_sentences": 1 if any(len(sent) < 11 for sent in sentences) else 0,
        "presence_long_sentences": 1 if any(len(sent) > 34 for sent in sentences) else 0,
        "presence_numbers": 1 if any(char.isdigit() for char in paragraph) else 0,
        "presence_more_capitals": 1 if sum(1 for c in paragraph if c.isupper()) > paragraph.count('.') * 2 else 0,
        "presence_although": 1 if "although" in words else 0,
        "presence_however": 1 if "however" in words else 0,
        "presence_but": 1 if "but" in words else 0,
        "presence_because": 1 if "because" in words else 0,
        "presence_this": 1 if "this" in words else 0,
        "presence_others_researchers": 1 if "others" in words or "researchers" in words else 0,
        "presence_et": 1 if "et" in words else 0
    }
    return features

# Function to read files and extract paragraphs
def extract_paragraphs_from_files(folder):
    paragraphs = []
    # List all files in the directory
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        # Check if it is a file
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
                # Split text into paragraphs
                paras = text.split('\n\n')
                # Filter out any empty paragraphs
                paras = [para for para in paras if para and not para.isspace()]
                paragraphs.extend(paras)
    return paragraphs
    

# Extract paragraphs from both folders
gpt_paragraphs = extract_paragraphs_from_files('gpt')
human_paragraphs = extract_paragraphs_from_files('human')

# Label the paragraphs (1 for GPT, 0 for Human)
gpt_labels = [1] * len(gpt_paragraphs)
human_labels = [0] * len(human_paragraphs)

# Combine the paragraphs and labels
paragraphs = gpt_paragraphs + human_paragraphs
labels = gpt_labels + human_labels

# Extract features for each paragraph
features = [extract_features(para) for para in paragraphs]

# Convert the features into a DataFrame
df_features = pd.DataFrame(features)
df_labels = pd.Series(labels)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)

def print_dataset_info(X_train, X_test, y_train, y_test):
    print("Features in X_train:", X_train.columns.tolist())
    print("Features in X_test:", X_test.columns.tolist())
    print("Name of y_train:", y_train.name)
    print("Name of y_test:", y_test.name)

# Calling the function with the mock data
print_dataset_info(X_train, X_test, y_train, y_test)

# Initialize and train the XGBoost classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = xgb_model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Return the accuracy
#accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")
print_dataset_info(X_train, X_test, y_train, y_test)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ... [rest of your code] ...

# After predicting on the test set
y_pred = xgb_model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print dataset info
print_dataset_info(X_train, X_test, y_train, y_test)

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'GPT'], yticklabels=['Human', 'GPT'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

# Compare actual and predicted in a DataFrame
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Plot comparison
comparison_df.reset_index(drop=True, inplace=True)
comparison_df.plot(kind='bar', figsize=(20, 5))
plt.title('Comparison of Actual and Predicted Labels')
plt.xlabel('Sample Index')
plt.ylabel('Label')
plt.legend()
plt.tight_layout()
plt.show()
