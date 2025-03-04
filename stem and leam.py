import nltk
import pandas as pd
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

df = pd.read_csv("/sms_spam.csv")  
print("Columns in CSV:", df.columns)

if "Text" not in df.columns:
    raise ValueError("CSV file must have a column named 'Text'!")

df = df.dropna(subset=["Text"])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def process_text(text):
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in words]
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words), ' '.join(stemmed_words), ' '.join(lemmatized_words), len(words)

df["Cleaned_Text"] = df["Text"].apply(clean_text)
df[["Original_Words", "Stemmed_Words", "Lemmatized_Words", "Word_Count"]] = df["Cleaned_Text"].apply(lambda x: pd.Series(process_text(x)))

df.to_csv("processed_sms_spam.csv", index=False)

all_words = ' '.join(df["Cleaned_Text"]).split()
word_freq = Counter(all_words)

print("\nTop 10 Most Frequent Words:")
for word, freq in word_freq.most_common(10):
    print(f"{word}: {freq}")

print(f"\nTotal Words in Dataset: {df['Word_Count'].sum()}")
print(f"Average Words Per Message: {df['Word_Count'].mean():.2f}")

all_filtered_words = [word for word in all_words if word not in stop_words]
filtered_word_freq = Counter(all_filtered_words)

print("\nTop 10 Most Frequent Words (Excluding Stopwords):")
for word, freq in filtered_word_freq.most_common(10):
    print(f"{word}: {freq}")

print("\nProcessing complete! The file 'processed_sms_spam.csv' is ready.")
