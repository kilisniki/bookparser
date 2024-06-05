import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import openpyxl
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from kn_irregular_verbs import get_irregular_verbs

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Load Spacy model
nlp = spacy.load('en_core_web_sm')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Shared variable to track progress
progress = {'total': 0, 'processed': 0, 'words_processed': 0}

# Define default dictionary factory as a top-level function
def default_dict_factory():
    return {
        'count': 0,
        'forms': set(),
        'sentences': set(),
        'sentence_forms': defaultdict(set),
        'chosen_sentences': [],
        'status': "Unknown"
    }

def remove_line_breaks(text):
    return re.sub(r'[\n\r\v\f]+', ' ', text)

# Function to clean and tokenize text into words
def extract_words_from_text(text):
    words = word_tokenize(text)
    return [word for word in words if word.isalpha()]

# Function to check if a word is a name
def is_name(word):
    doc = nlp(word)
    for ent in doc.ents:
        if ent.text == word and ent.label_ in ['PERSON', 'GPE', 'ORG']:
            return True
    return False

# Function to get the WordNet POS tag for lemmatization
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': 'a', 'N': 'n', 'V': 'v', 'R': 'r'}
    return tag_dict.get(tag, 'n')

# Function to count words in a chunk of text
def count_words_in_chunk(chunk):
    global progress
    word_map = defaultdict(default_dict_factory)
    for word in chunk:
        if not is_name(word):
            pos = get_wordnet_pos(word)
            lemma = lemmatizer.lemmatize(word, pos)  # Use POS tagging for lemmatization
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
            lowercase_word = word.lower()
            final_word_map[lowercase_word]['count'] += data['count']
            for form in data['forms']:
                final_word_map[lowercase_word]['forms'].add(form.lower())

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

# Function to normalize and zip words
def normalize_and_zip(word_map):
    normalized_map = defaultdict(default_dict_factory)
    irregular_verbs = get_irregular_verbs()

    for word, data in word_map.items():
        lemma = lemmatizer.lemmatize(word, get_wordnet_pos(word))

        # Check if the word is an irregular verb
        for base, forms in irregular_verbs.items():
            if word in forms or word == base:
                lemma = base
                break

        normalized_map[lemma]['count'] += data['count']
        normalized_map[lemma]['forms'].update(data['forms'])
        normalized_map[lemma]['sentences'].update(data['sentences'])

    return normalized_map

# Function to choose median sentences
# def choose_median_sentences(sentences):
#     sorted_sentences = sorted(sentences, key=len)
#     mid_index = len(sorted_sentences) // 2
#     start_index = max(0, mid_index - 1)
#     end_index = min(len(sorted_sentences), mid_index + 1)
#     return sorted_sentences[start_index:end_index]

# Function to find a sentence for each word
def find_sentences_for_words(word_map, text):
    sentences = sent_tokenize(text)
    for sentence in sentences:
        words_in_sentence = extract_words_from_text(sentence)
        for word in words_in_sentence:
            lowercase_word = word.lower()
            lemma = lemmatizer.lemmatize(lowercase_word, get_wordnet_pos(lowercase_word))
            # Check both the original word and the lemma in lowercase
            for key in word_map:
                for form in word_map[key]['forms']:
                    if lowercase_word == form or lemma == form:
                        word_map[key]['sentence_forms'][form].add(sentence)
                        break
    return word_map

# Define the function to choose median sentences (you can implement this based on your specific needs)
def choose_median_sentences(sentences):
    # Placeholder implementation: return the middle sentence or the first one if there's only one
    if not sentences:
        return []
    sorted_sentences = sorted(sentences)
    mid_index = len(sorted_sentences) // 2
    return [sorted_sentences[mid_index]] if len(sorted_sentences) % 2 == 1 else sorted_sentences[mid_index-1:mid_index+1]

def update_word_map_with_sentences(word_map):
    for word, data in word_map.items():
        if 'sentence_forms' in data:
            chosen_sentences = []
            for form, sentences in data['sentence_forms'].items():
                all_sentences = list(sentences)
                chosen_sentences.extend(choose_median_sentences(all_sentences))
            word_map[word]['chosen_sentences'] = chosen_sentences
    return word_map

def enrich_word_map(word_map, vocabulary):
    for word in word_map:
        if word in vocabulary and vocabulary[word]['status'] == 'Known':
            word_map[word]['status'] = 'Known'
    return word_map

# Function to execute the main script
# This function should
def process_text_file(input_filename, vocabulary):
    # Load the book text
    print("Loading book text...")
    with open(input_filename, 'r', encoding='utf-8') as f:
        text = f.read()
    print("Book text loaded.")

    # Extract and clean words
    print("Extracting and cleaning words...")
    text = remove_line_breaks(text)
    words = extract_words_from_text(text)
    print("Words extracted and cleaned.")

    # Count words in the text
    print("Counting words in text...")
    word_map = count_words(words)
    print("Word counting completed.")

    # Normalize and zip words
    print("Normalizing and zipping words...")
    normalized_map = normalize_and_zip(word_map)
    print("Words normalized and zipped.")

    # Find sentences for each word
    print("Finding sentences for each word...")
    final_word_map = find_sentences_for_words(normalized_map, text)
    print("Sentences found for each word.")

    if (vocabulary):
        print("Enrich by user's vocabulary")
        enrich_word_map(final_word_map, vocabulary)
        print("Words enriched by vocabulary")

    print("Updating word map with chosen sentences...")
    return update_word_map_with_sentences(final_word_map)
