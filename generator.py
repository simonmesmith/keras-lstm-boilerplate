from __future__ import print_function
import helper
import modeller
import numpy as np
import random
import sys

def write(inputFilename, outputFilename, numberOfOutputs, creativity):

    text = helper.getText(inputFilename)
    chars = helper.getChars(text)
    char_indices = helper.getCharIndices(chars)
    indices_char = helper.getIndicesChar(chars)
    maxlen = helper.getMaxLen(text)
    model = modeller.getModel(inputFilename)

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
    outputs = []
    for iteration in range(1, numberOfOutputs): # For iterations in the range specified...

        # Get a random seed.
        start_index = random.randint(0, len(text) - maxlen - 1) # Sets the start index for the seed to a random start point with sufficient length remaining for an appropriate-length seed
        seed = text[start_index: start_index + maxlen] # Gets a string of maxlen length from the text using the random start index variable

        # Set a variable to hold the generated text and set it to the seed to start.
        generated = '' # Sets variable for generated name
        generated += seed # Sets the generated variable to the seed to start

        # Loop and generate text.
        for i in range(maxlen): # Looping through maxlen times as a test; seems logical, since this should be related to the length of strings in the input text

            # Vectorize the seed.
            # To learn: Why do we set x[] = 1? Why does that 1 have a period following it?
            x = np.zeros((1, maxlen, len(chars))) # Set a variable to an array of the specified shape that will hold the name seed in vectorized form
            for t, char in enumerate(seed): # Loop through each character in the seed, with t as the character index and char as the character
                x[0, t, char_indices[char]] = 1. # Append the character index from the seed (t) and the character index of the character to the variable x

            # Predict the next character for the  seed.
            # To learn: Why do we use sample(preds, creativity)? Assumption: Preds returns an array of likely next characters, with some having a higher probability
            # than others, and a lower creativity setting selects a more probable character whereas a higher creativity setting selects a less probable (i.e. more
            # creative) character. Is this true?
            preds = model.predict(x, verbose=0)[0] # Set a variable to hold a prediction; x is the input data, verbosity of 0 to suppress logging
            next_index = sample(preds, creativity) # Take a random sample of the prediction using the creativity value
            next_char = indices_char[next_index] # Set a next_char variable to an actual character using the inverse character indices variable

            # Update generated and seed variables.
            generated += next_char # Add the next character to the generated variable
            seed = seed[1:] + next_char # Remove first letter of seed and add next_char at the end; this is the seed used to predict the next character in the next loop

        # Append the generated text to the output
        outputs.append(generated)

    # Save outputs to file.
    text_file = open('outputs/' + outputFilename, "w")
    text_file.write('\n'.join(outputs))
    text_file.close()
