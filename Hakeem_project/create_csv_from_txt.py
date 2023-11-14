import pandas as pd
import nltk

nltk.download('punkt')  # Download NLTK data if not already downloaded

from nltk.tokenize import blankline_tokenize

def text_to_csv(text, file_name):
    # Split the text into paragraphs using NLTK
    paragraphs = blankline_tokenize(text)
    # Filter out any empty paragraphs
    paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]
    # Create a DataFrame with one column named 'text'
    df = pd.DataFrame(paragraphs, columns=['text'])
    # Save the DataFrame to a CSV file
    df.to_csv(file_name, index=False)

# Function to read text from a file
def read_text_from_file(file_name):
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")
        return None

# Input file name
input_file = 'test.txt'

# Read text from the input file
text = read_text_from_file(input_file)

if text:
    # Output file name for the CSV
    output_csv = 'output.csv'

    # Convert text to CSV
    text_to_csv(text, output_csv)

    print(f"Text from '{input_file}' has been converted to '{output_csv}'.")

output_csv = f'{input_file}_output.csv'

# Given the file has been uploaded, I'll rewrite the code without the nltk download and assuming the file is present.

import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import numpy as np

# Ensure that NLTK's tokenizers and other resources are available
# nltk.download('punkt') - this line is not needed as nltk data is already available in the environment

column_descriptions = {
    "text": "The original paragraph of text from the dataset.",
    "num_sentences": "The number of sentences in the paragraph.",
    "num_words": "The total number of words in the paragraph.",
    "presence_parentheses": "Indicates the presence (1) or absence (0) of parentheses in the paragraph.",
    "presence_dash": "Indicates the presence (1) or absence (0) of a dash (-) in the paragraph.",
    "presence_semicolon_colon": "Indicates the presence (1) or absence (0) of a semicolon (;) or colon (:) in the paragraph.",
    "presence_question_mark": "Indicates the presence (1) or absence (0) of a question mark (?) in the paragraph.",
    "presence_apostrophe": "Indicates the presence (1) or absence (0) of an apostrophe (') in the paragraph.",
    "std_sentence_length": "The standard deviation of the lengths of the sentences in the paragraph.",
    "mean_diff_sentence_length": "The mean difference in length between consecutive sentences in the paragraph.",
    "presence_short_sentences": "Indicates if there is at least one sentence with fewer than 11 words (1) or not (0) in the paragraph.",
    "presence_long_sentences": "Indicates if there is at least one sentence with more than 34 words (1) or not (0) in the paragraph.",
    "presence_numbers": "Indicates the presence (1) or absence (0) of numbers in the paragraph.",
    "presence_more_capitals": "Indicates if there are more than twice as many capital letters as periods (1) or not (0) in the paragraph.",
    "presence_although": "Indicates the presence (1) or absence (0) of the word 'although' in the paragraph.",
    "presence_however": "Indicates the presence (1) or absence (0) of the word 'however' in the paragraph.",
    "presence_but": "Indicates the presence (1) or absence (0) of the word 'but' in the paragraph.",
    "presence_because": "Indicates the presence (1) or absence (0) of the word 'because' in the paragraph.",
    "presence_this": "Indicates the presence (1) or absence (0) of the word 'this' in the paragraph.",
    "presence_others_researchers": "Indicates the presence (1) or absence (0) of the words 'others' or 'researchers' in the paragraph.",
    "presence_et": "Indicates the presence (1) or absence (0) of the abbreviation 'et' in the paragraph."
}


# Define the function to extract features from a paragraph
def extract_features(paragraph):
    sentences = sent_tokenize(paragraph)
    words = word_tokenize(paragraph)
    
    # Features to be calculated
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

# Load the dataset from the uploaded file
df = pd.read_csv('output.csv')

# Apply the extract_features function to the 'text' column
features_df = df['text'].apply(lambda para: pd.Series(extract_features(para)))

# Concatenate the original DataFrame with the new features DataFrame
enhanced_df = pd.concat([df, features_df], axis=1)

# Save the enhanced dataframe to a new CSV file
#output_file_path = 'human_enhanced_output.csv'
output_file_path = output_csv
enhanced_df.to_csv(output_file_path, index=False)

# Display the path to the new CSV file
output_file_path, enhanced_df.head()

