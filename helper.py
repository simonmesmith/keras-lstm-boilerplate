import string

# Get text and lowercase it. (Assumption: Lowercase to reduce the number of unique characters.)
def getText(inputFilename):
    return open('inputs/' + inputFilename).read().lower()

# Get unique characters from the text in list format in ascending order.
def getChars(text):
    return sorted(list(set(text)))

# Create a character index array for the unique characters (e.g. ['a': 0, 'b': 1]).
def getCharIndices(chars):
    return dict((c, i) for i, c in enumerate(chars))

# Create another character index array that's the inverse of char_indices (e.g. ['0': a, '1': b]).
def getIndicesChar(chars):
    return dict((i, c) for i, c in enumerate(chars))

# Set the maximum length of a name (used throughout the code). Here set to the average length of a drug name.
def getMaxLen(text):
    num_names = len(text.splitlines()) # Number of names (lines in text)
    num_chars = len(text) # Total number of characters (includes white space and line breaks; to do: remove white space and line breaks)
    return int(round(num_chars/num_names)) # Set the maxlen value to the average length of a drug brand name

def cleanGenerated(text):
    splitText = string.split(text, '\n')
    return splitText[1]
