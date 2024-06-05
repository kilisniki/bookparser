from kn_parser import process_text_file
from kn_save_to_xlsx import save_to_excel
from kn_parse_from_xlsx import parse_vocabulary
from kn_merge_word_maps import merge_vocabularies
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python kn_main.py <book> <your_vocabulary>")
        sys.exit(1)

    input_filename = sys.argv[1]
    vocabulary_filename = sys.argv[2]

    if (vocabulary_filename):
	    vocabulary = parse_vocabulary(vocabulary_filename)

    parsedData = process_text_file(input_filename, vocabulary)
    save_to_excel(parsedData, 'book_vocabulary.xlsx')

    mergedData = merge_vocabularies(vocabulary, parsedData)
    save_to_excel(mergedData, 'my_vocabulary.xlsx')
