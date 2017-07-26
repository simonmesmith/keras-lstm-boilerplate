from __future__ import print_function
import numpy as np
import model as nameModel
import random
import sys

# Get text and lowercase it. (Assumption: Lowercase to reduce the number of unique characters.)
text = open('real-drug-brand-names.txt').read().lower()

# Get unique characters from the text in list format in ascending order.
chars = sorted(list(set(text)))

# Create a character index array for the unique characters (e.g. ['a': 0, 'b': 1]).
char_indices = dict((c, i) for i, c in enumerate(chars))

# Create another character index array that's the inverse of char_indices (e.g. ['0': a, '1': b]).
indices_char = dict((i, c) for i, c in enumerate(chars))

# Set the maximum length of a name (used throughout the code). Here set to the average length of a drug name.
num_names = len(text.splitlines()) # Number of names (lines in text)
num_chars = len(text) # Total number of characters (includes white space and line breaks; to do: remove white space and line breaks)
maxlen = int(round(num_chars/num_names)) # Set the maxlen value to the average length of a drug brand name

model = nameModel.get(text, chars, char_indices, indices_char, maxlen)

# Create a function that "samples an index from a probability array," according to the author.
# To learn: What does this do? Why do we need it?
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64') # Converts the variable preds to an array of type float64
    preds = np.log(preds) / temperature # Converts the variable preds to a logarithm of preds divided by the temperature
    exp_preds = np.exp(preds) # Sets a variable to hold the exponential of preds
    preds = exp_preds / np.sum(exp_preds) # Sums array elements in the variable exp_preds
    probas = np.random.multinomial(1, preds, 1) # Sets a variable to sample from a multinomial distribution
    return np.argmax(probas) # "Returns the indices of the maximum values along an axis"; don't understand this fully

# Make a batch of new drug names
for iteration in range(1, 10): # For iterations in the range specified...

    # Print header.
    print() # Prints blank
    print('-' * 50) # Prints a row of hyphens
    print('Iteration', iteration) # Prints "Iteration X"

    # Set a random start index variable to use for getting a string of text to seed predictions.
    start_index = random.randint(0, len(text) - maxlen - 1)

    # Loop through diversity values and generate strings.
    # To learn: Are these diversity values effectively for "creativity?" Why are they the values they are?
    for diversity in [0.2, 0.5, 1.0, 1.2]:

        # Print a header for each diversity run.
        print() # Prints blank
        print('----- diversity:', diversity) # Prints  diversity

        # Create a variable for the generated name and get a string to use as the seed.
        # To learn: Why do we use sys.stdout.write() below instead of print()?
        generated = '' # Sets variable for generated name
        name = text[start_index: start_index + maxlen] # Gets a string of maxlen length from the text using the random start index variable
        generated += name # Sets the generated variable to the name
        print('----- seed:', name) # Prints name seed
        sys.stdout.write(generated) # To learn: Why is this here versus below? What actually happens here? Gets written to buffer? Since generated set to nothing above? What happens if I remove this?

        # Loop and generate text.
        # To learn: What's the significance of doing this 400 times?
        for i in range(400):

            # Vectorize the name seed.
            # To learn: Why do we set x[] = 1? Why does that 1 have a period following it?
            x = np.zeros((1, maxlen, len(chars))) # Set a variable to an array of the specified shape that will hold the name seed in vectorized form
            for t, char in enumerate(name): # Loop through each character in the name, with t as the character index and char as the character
                x[0, t, char_indices[char]] = 1. # Append the character index from the name (t) and the character index of the character to the variable x

            # Predict the next character for the random name seed.
            # To learn: Why do we use sample(preds, diversity)? Assumption: Preds returns an array of likely next characters, with some having a higher probability
            # than others, and a lower diversity setting selects a more probable character whereas a higher diversity setting selects a less probable (i.e. more
            # creative) character. Is this true?
            preds = model.predict(x, verbose=0)[0] # Set a variable to hold a prediction; x is the input data, verbosity of 0 to suppress logging
            next_index = sample(preds, diversity) # Take a random sample of the prediction using the diversity value
            next_char = indices_char[next_index] # Set a next_char variable to an actual character using the inverse character indices variable

            generated += next_char # Add the next character to the generated variable
            name = name[1:] + next_char # Adds the next_char to the name? To learn: What's going on here? Ah, I think it's adding the next character to the original name and then having the system predict against that name on the next loop

            sys.stdout.write(next_char) # To learn: Is this getting buffered here? I think it may be writing each character to the screen; true?
            sys.stdout.flush() # To learn: What's getting flushed here? The buffered next_char written to sys.stdout.write() above?

        print() # To learn: Just printing a blank line here, or somehow helping get the generated text flushed?
