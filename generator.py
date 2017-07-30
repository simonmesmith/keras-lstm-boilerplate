from __future__ import print_function
import helper
import modeller
import numpy as np
import random
import sys

def write(inputFilename, outputFilename, scanLength=5, outputLength=10, numberOfOutputs=10, creativity=0.2):

    text = helper.getText(inputFilename)
    chars = helper.getChars(text)
    char_indices = helper.getCharIndices(chars)
    indices_char = helper.getIndicesChar(chars)
    model = modeller.getModel(inputFilename, scanLength)

    # Create a function that "samples an index from a probability array," according to the author.
    # To learn: What does this do? Why do we need it?
    def sample(preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64') # Converts the variable preds to an array of type float64
        preds = np.log(preds) / temperature # Converts the variable preds to a logarithm of preds divided by the temperature
        exp_preds = np.exp(preds) # Sets a variable to hold the exponential of preds
        preds = exp_preds / np.sum(exp_preds) # Sums array elements in the variable exp_preds
        probas = np.random.multinomial(1, preds, 1) # Sets a variable to sample from a multinomial distribution
        return np.argmax(probas) # "Returns the indices of the maximum values along an axis"; don't understand this fully

    # Make a batch of new outputs
    outputs = []
    for iteration in range(1, numberOfOutputs): # For iterations in the range specified...

        # Create a generated variable and start it off with seed text.
        start_index = random.randint(0, len(text) - scanLength - 1) # Sets the start index for the seed to a random start point with sufficient length remaining for an appropriate-length seed
        seed = text[start_index: start_index + scanLength] # Gets a string of scanLength length from the text using the random start index variable
        generated = seed # Sets a generated variable using the seed as the starting text

        # Loop and generate text.
        for i in range(outputLength): # Create an output of outputLength length by looping that many times and adding a character each time

            # Vectorize the generated text.
            # To learn: Why do we set x[] = 1? Why does that 1 have a period following it?
            x = np.zeros((1, scanLength, len(chars))) # Set a variable to an array of the specified shape that will hold the generated text in vectorized form
            for t, char in enumerate(generated): # Loop through each character in the generated text, with t as the character index and char as the character
                x[0, t, char_indices[char]] = 1. # Append the character index from the generated text (t) and the character index of the character to the variable x

            # Predict the next character for the generated text.
            # To learn: Why do we use sample(preds, creativity)? Assumption: Preds returns an array of likely next characters, with some having a higher probability
            # than others, and a lower creativity setting selects a more probable character whereas a higher creativity setting selects a less probable (i.e. more
            # creative) character. Is this true?
            preds = model.predict(x, verbose=0)[0] # Set a variable to hold a prediction; x is the input data, verbosity of 0 to suppress logging
            next_index = sample(preds, creativity) # Take a random sample of the prediction using the creativity value
            next_char = indices_char[next_index] # Set a next_char variable to an actual character using the inverse character indices variable

            # Update generated text. Note that we overwrite the text moving from left to right, therefore overwriting the initial seed text with all generated text.
            generated = seed[1:] + next_char # Remove first letter of seed and add next_char at the end; this is the seed used to predict the next character in the next loop

        # Append the generated text to the output
        outputs.append(generated)

    # Save outputs to file.
    text_file = open('outputs/' + outputFilename, "w")
    text_file.write('\n'.join(outputs))
    text_file.close()
