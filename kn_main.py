from kn_parser import process_text_file
from kn_save_to_xlsx import save_to_excel
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python kn_main.py <input_filename> <output_filename>")
        sys.exit(1)

    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    parsedData = process_text_file(input_filename)
    print("Word map updated with chosen sentences.")
    save_to_excel(parsedData, output_filename)
