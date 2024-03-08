import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import dotenv

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def parse_pdf(pdf_path):
    text_content = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num in range(len(pdf.pages)):
            page = pdf.pages[page_num]
            text_content += page.extract_text()
    return text_content

def segment_text(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

def tokenize_sentences(sentences):
    tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
    return tokens

def remove_stop_words(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return filtered_tokens

def stem_words(tokens):
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

def lemmatize_words(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

# Example usage
pdf_path = 'pdftoParse.pdf'  # Replace with the actual path to your PDF file

try:
    parsed_text = parse_pdf(pdf_path)
    print("Parsed Text:")
    print(parsed_text)

    segmented_text = segment_text(parsed_text)
    print("\nSegmented Text:")
    for sentence in segmented_text:
        print(sentence)

    tokenized_sentences = tokenize_sentences(segmented_text)
    print("\nTokenized Sentences:")
    for tokens in tokenized_sentences:
        print(tokens)

    filtered_tokens = remove_stop_words(tokenized_sentences[0])
    print("\nTokens after Stop Words Removal:")
    print(filtered_tokens)

    stemmed_tokens = stem_words(filtered_tokens)
    print("\nStemmed Tokens:")
    print(stemmed_tokens)

    lemmatized_tokens = lemmatize_words(filtered_tokens)
    print("\nLemmatized Tokens:")
    print(lemmatized_tokens)

except Exception as e:
    print(f"An error occurred: {e}")
