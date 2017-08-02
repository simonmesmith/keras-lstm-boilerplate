import string

# Get text and lowercase it. (Lowercase to reduce the number of unique characters.)
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

# Clean the output (such as to strip leading or trailing characters not overwritten cleanly as we generate strings from seeds).
def cleanOutput(outputFilename, minOutputLength=3):
    cleaned = []
    with open('outputs/' + outputFilename) as f:
        for line in f:
            if (len(line.strip()) > minOutputLength):
                cleaned.append(line.strip())
    outputFile = open('outputs/' + outputFilename, "w")
    outputFile.write('\n'.join(cleaned))
    outputFile.close()
