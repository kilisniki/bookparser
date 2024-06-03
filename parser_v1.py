import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import openpyxl
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import threading
import time

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Shared variable to track progress
progress = {'total': 0, 'processed': 0, 'words_processed': 0}

# Define default dictionary factory as a top-level function
def default_dict_factory():
    return {'count': 0, 'forms': set(), 'sentences': set()}

# Function to clean and tokenize text into words
def extract_words_from_text(text):
    cleaned_text = re.sub(r'[.,\/#!$%\^&\*;:{}=\-_`~()?"“”‘’]', '', text)
    words = word_tokenize(cleaned_text)
    return [word for word in words if word.isalpha()]

# Function to check if a word is a name
def is_name(word):
    doc = nlp(word)
    for ent in doc.ents:
        if ent.text == word and ent.label_ in ['PERSON', 'GPE', 'ORG']:
            return True
    return False

# Function to count words in a chunk of text
def count_words_in_chunk(chunk):
    global progress
    word_map = defaultdict(default_dict_factory)
    for word in chunk:
        if not is_name(word):
            lemma = lemmatizer.lemmatize(word.lower())  # Convert to lowercase here
            word_map[lemma]['count'] += 1
            word_map[lemma]['forms'].add(word)
            progress['words_processed'] += 1
    progress['processed'] += 1
    return word_map

# Function to merge word maps
def merge_word_maps(word_maps):
    final_word_map = defaultdict(default_dict_factory)
    for word_map in word_maps:
        for word, data in word_map.items():
            final_word_map[word]['count'] += data['count']
            final_word_map[word]['forms'].update(data['forms'])
    return final_word_map

# Function to count words in the entire text using parallel processing
def count_words(words, num_chunks=10):
    global progress
    # Split words into chunks
    chunk_size = len(words) // num_chunks
    chunks = [words[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    if len(words) % num_chunks:
        chunks.append(words[num_chunks*chunk_size:])
    progress['total'] = len(chunks)

    # Process chunks in parallel
    with ProcessPoolExecutor() as executor:
        word_maps = list(executor.map(count_words_in_chunk, chunks))

    # Merge word maps from all chunks
    final_word_map = merge_word_maps(word_maps)

    return final_word_map

# Function to save words to an Excel file
def save_to_excel(word_map, filename):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['Word', 'Count', 'Forms', 'Sentences'])
    for word, data in word_map.items():
        # Sort sentences and choose the middle ones
        sorted_sentences = sorted(data['sentences'], key=len)
        mid_index = len(sorted_sentences) // 2
        sentences_to_include = sorted_sentences[max(0, mid_index-1):mid_index+1]
        ws.append([word, data['count'], ', '.join(data['forms']), '. '.join(sentences_to_include)])
    wb.save(filename)

# Function to normalize and zip words
def normalize_and_zip(word_map):
    normalized_map = defaultdict(default_dict_factory)
    for word, data in word_map.items():
        lemma = lemmatizer.lemmatize(word)
        normalized_map[lemma]['count'] += data['count']
        normalized_map[lemma]['forms'].update(data['forms'])
    return normalized_map

# Function to find a sentence for each word
def find_sentences_for_words(word_map, text):
    sentences = sent_tokenize(text)
    for sentence in sentences:
        words_in_sentence = extract_words_from_text(sentence)
        for word in words_in_sentence:
            lemma = lemmatizer.lemmatize(word.lower())
            if lemma in word_map:
                word_map[lemma]['sentences'].add(sentence)
    return word_map

# Function to print progress
def print_progress():
    while progress['processed'] < progress['total']:
        percentage = (progress['processed'] / progress['total']) * 100
        print(f"Processed {progress['processed']} out of {progress['total']} chunks ({percentage:.2f}%). Words processed: {progress['words_processed']}")
        time.sleep(3)

# Main function
def main():
    # Load the book text
    print("Loading book text...")
    with open('thewitcher.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print("Book text loaded.")

    # Extract and clean words
    print("Extracting and cleaning words...")
    words = extract_words_from_text(text)
    print("Words extracted and cleaned.")

    # Start progress thread
    print("Starting progress thread...")
    progress_thread = threading.Thread(target=print_progress)
    progress_thread.start()

    # Count words in the text
    print("Counting words in text...")
    word_map = count_words(words)
    print("Word counting completed.")

    # Ensure progress thread completes
    progress_thread.join()
    print("Progress thread completed.")

    # Normalize and zip words
    print("Normalizing and zipping words...")
    normalized_map = normalize_and_zip(word_map)
    print("Words normalized and zipped.")

    # Find sentences for each word
    print("Finding sentences for each word...")
    final_word_map = find_sentences_for_words(normalized_map, text)
    print("Sentences found for each word.")

    # Save words to an Excel file
    print("Saving word counts to Excel...")
    save_to_excel(final_word_map, 'normalized_word_counts.xlsx')
    print("Word counts saved to Excel.")

if __name__ == '__main__':
    main()
