import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import openpyxl
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

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
    return {'count': 0, 'forms': set(), 'sentences': set()}

def remove_line_breaks(text):
    return re.sub(r'[\n\r\v\f]+', ' ', text)

# Function to clean and tokenize text into words
def extract_words_from_text(text):
    text = text.lower()  # Convert text to lowercase
    cleaned_text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(cleaned_text)
    return [word for word in words if word.isalpha()]

# Function to check if a word is a name
def is_name(word):
    doc = nlp(word)
    for ent in doc.ents:
        if ent.text.lower() == word.lower() and ent.label_ in ['PERSON', 'GPE', 'ORG']:
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

# Function to choose median sentences
def choose_median_sentences(sentences):
    sorted_sentences = sorted(sentences, key=len)
    mid_index = len(sorted_sentences) // 2
    start_index = max(0, mid_index - 1)
    end_index = min(len(sorted_sentences), mid_index + 1)
    return sorted_sentences[start_index:end_index]

# Function to save words to an Excel file
def save_to_excel(word_map, filename):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['Word', 'Count', 'Forms', 'Sentences'])
    for word, data in word_map.items():
        # Aggregate all sentences for the word
        all_sentences = list(data['sentences'])
        chosen_sentences = choose_median_sentences(all_sentences)
        ws.append([
            word,
            data['count'],
            ', '.join(data['forms']),
            ' | '.join(chosen_sentences)  # Use ' | ' to separate sentences
        ])
    wb.save(filename)

# Function to normalize and zip words
def normalize_and_zip(word_map):
    normalized_map = defaultdict(default_dict_factory)
    irregular_verbs = {
        'arise': ['arose', 'arisen'],
        'awake': ['awoke', 'awoken'],
        'be': ['am', 'is', 'are', 'was', 'were', 'being', 'been'],
        'bear': ['bore', 'borne'],
        'beat': ['beat', 'beaten'],
        'become': ['became', 'become'],
        'begin': ['began', 'begun'],
        'bend': ['bent'],
        'bet': ['bet'],
        'bind': ['bound'],
        'bite': ['bit', 'bitten'],
        'bleed': ['bled'],
        'blow': ['blew', 'blown'],
        'break': ['broke', 'broken'],
        'bring': ['brought'],
        'build': ['built'],
        'burn': ['burnt', 'burned'],
        'burst': ['burst'],
        'buy': ['bought'],
        'catch': ['caught'],
        'choose': ['chose', 'chosen'],
        'come': ['came', 'come'],
        'cost': ['cost'],
        'cut': ['cut'],
        'deal': ['dealt'],
        'dig': ['dug'],
        'do': ['did', 'done', 'doing'],
        'draw': ['drew', 'drawn'],
        'dream': ['dreamt', 'dreamed'],
        'drink': ['drank', 'drunk'],
        'drive': ['drove', 'driven'],
        'eat': ['ate', 'eaten'],
        'fall': ['fell', 'fallen'],
        'feed': ['fed'],
        'feel': ['felt'],
        'fight': ['fought'],
        'find': ['found'],
        'fly': ['flew', 'flown'],
        'forget': ['forgot', 'forgotten'],
        'forgive': ['forgave', 'forgiven'],
        'freeze': ['froze', 'frozen'],
        'get': ['got', 'gotten'],
        'give': ['gave', 'given'],
        'go': ['went', 'gone', 'going'],
        'grow': ['grew', 'grown'],
        'hang': ['hung'],
        'have': ['had'],
        'hear': ['heard'],
        'hide': ['hid', 'hidden'],
        'hit': ['hit'],
        'hold': ['held'],
        'hurt': ['hurt'],
        'keep': ['kept'],
        'know': ['knew', 'known'],
        'lay': ['laid'],
        'lead': ['led'],
        'leave': ['left'],
        'lend': ['lent'],
        'let': ['let'],
        'lie': ['lay', 'lain'],
        'lose': ['lost'],
        'make': ['made'],
        'mean': ['meant'],
        'meet': ['met'],
        'pay': ['paid'],
        'put': ['put'],
        'read': ['read'],
        'ride': ['rode', 'ridden'],
        'ring': ['rang', 'rung'],
        'rise': ['rose', 'risen'],
        'run': ['ran', 'run'],
        'say': ['said'],
        'see': ['saw', 'seen'],
        'sell': ['sold'],
        'send': ['sent'],
        'set': ['set'],
        'shake': ['shook', 'shaken'],
        'shine': ['shone'],
        'shoot': ['shot'],
        'show': ['showed', 'shown'],
        'shut': ['shut'],
        'sing': ['sang', 'sung'],
        'sink': ['sank', 'sunk'],
        'sit': ['sat'],
        'sleep': ['slept'],
        'speak': ['spoke', 'spoken'],
        'spend': ['spent'],
        'stand': ['stood'],
        'steal': ['stole', 'stolen'],
        'swim': ['swam', 'swum'],
        'take': ['took', 'taken'],
        'teach': ['taught'],
        'tear': ['tore', 'torn'],
        'tell': ['told'],
        'think': ['thought'],
        'throw': ['threw', 'thrown'],
        'understand': ['understood'],
        'wake': ['woke', 'woken'],
        'wear': ['wore', 'worn'],
        'win': ['won'],
        'write': ['wrote', 'written'],
        # Add more irregular verbs as needed
    }

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

# Function to find a sentence for each word
def find_sentences_for_words(word_map, text):
    sentences = sent_tokenize(text)
    for sentence in sentences:
        words_in_sentence = extract_words_from_text(sentence)
        for word in words_in_sentence:
            lemma = lemmatizer.lemmatize(word.lower(), get_wordnet_pos(word))
            if lemma in word_map:
                word_map[lemma]['sentences'].add(sentence)
    return word_map

# Main function
def main():
    # Load the book text
    print("Loading book text...")
    with open('thewitcher.txt', 'r', encoding='utf-8') as f:
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

    # Save words to an Excel file
    print("Saving word counts to Excel...")
    save_to_excel(final_word_map, 'normalized_word_counts.xlsx')
    print("Word counts saved to Excel.")

if __name__ == '__main__':
    main()
