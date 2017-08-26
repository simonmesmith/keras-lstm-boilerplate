import re
import string

# Get text and lowercase it. (Lowercase to reduce the number of unique characters.)
def get_text(input_filename):
    return open('inputs/' + input_filename).read().lower()

# Get unique characters from the text in list format in ascending order.
def get_chars(text):
    return sorted(list(set(text)))

# Create a character index array for the unique characters (e.g. ['a': 0, 'b': 1]).
def get_char_indices(chars):
    return dict((c, i) for i, c in enumerate(chars))

# Create another character index array that's the inverse of char_indices (e.g. ['0': a, '1': b]).
def get_indices_char(chars):
    return dict((i, c) for i, c in enumerate(chars))

# Clean the output (such as to strip leading or trailing characters not overwritten cleanly as we generate strings from seeds).
def clean_output(output_filename, min_output_length=3):
    cleaned = []
    with open('outputs/' + output_filename) as output:
        for text in output:
            for line in text.strip().split('\r'):
                if (len(line.strip()) > min_output_length):
                    cleaned.append(line.strip())
    output_file = open('outputs/' + output_filename, "w")
    output_file.write('\n'.join(cleaned))
    output_file.close()
