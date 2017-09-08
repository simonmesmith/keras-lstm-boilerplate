import re
import string

# Get text and lowercase it. (Lowercase to reduce the number of unique characters.)
def get_text(input_filename):
    return open('inputs/' + input_filename).read().lower()

# Get unique strings from the text in list format in ascending order.
def get_unique_strings(text, level):
    if level is 'word':
        unique_strings = set(text.split())
    else:
        unique_strings = set(text)
    return sorted(list(unique_strings))

# Create a string index array for the unique strings (e.g. ['a': 0, 'b': 1]).
def get_unique_string_indices(unique_strings):
    return dict((s, i) for i, s in enumerate(unique_strings))

# Create another string index array that's the inverse of unique_string_indices (e.g. [0: 'a', 1: 'b']).
def get_indices_unique_string(unique_strings):
    return dict((i, s) for i, s in enumerate(unique_strings))

# Return an array of words in their original order if the level is "words" otherwise just the text (which is the characters in order)
def split_strings_as_needed(strings, level):
    return strings.split() if level == 'word' else strings

def join_strings_as_needed(strings, level):
    return ' '.join(strings) if level == 'word' else strings

# Clean the output (such as to strip leading or trailing characters not overwritten cleanly as we generate strings from seeds).
def clean_output(output_filename, min_row_length=3):
    cleaned = []
    with open('outputs/' + output_filename) as output:
        for text in output:
            for line in text.strip().split('\r'):
                if (len(line.strip()) > min_row_length):
                    cleaned.append(line.strip())
    output_file = open('outputs/' + output_filename, "w")
    output_file.write('\n'.join(cleaned))
    output_file.close()
