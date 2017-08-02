from __future__ import print_function
import helper
import modeller
import numpy as np
import random
import sys

def write(inputFilename, outputFilename, scanLength=5, outputLength=10, numberOfOutputs=10, creativity=0.2):

    # Set key variables.
    text = helper.getText(inputFilename)
    chars = helper.getChars(text)
    char_indices = helper.getCharIndices(chars)
    indices_char = helper.getIndicesChar(chars)
    model = modeller.getModel(inputFilename, scanLength)

    # Create a function that returns a prediction from an array of predictions based on the specified "temperature."
    # This allows us to return a more or less creative (i.e. more or less probable) prediction.
    # To learn: Would like to have someone walk through each step of this function and what it's doing.
    def sample(predictions, temperature=1.0):
        predictions = np.asarray(predictions).astype('float64') # Converts the variable predictions to an array of type float64
        predictions = np.log(predictions) / temperature # Converts the variable predictions to a logarithm of predictions divided by the temperature
        exp_predictions = np.exp(predictions) # Sets a variable to hold the exponential of predictions
        predictions = exp_predictions / np.sum(exp_predictions) # Sums array elements in the variable exp_predictions
        probabilities = np.random.multinomial(1, predictions, 1) # Sets a variable to sample from a multinomial distribution
        return np.argmax(probabilities) # "Returns the indices of the maximum values along an axis"; don't understand this fully

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
            # To learn: Why do we set x[] = 1? Why does that 1 have a period following it? Assumption: one-hot encoding.
            x = np.zeros((1, scanLength, len(chars))) # Set a variable to an array of the specified shape that will hold the generated text in vectorized form
            for t, char in enumerate(generated): # Loop through each character in the generated text, with t as the character index and char as the character
                x[0, t, char_indices[char]] = 1. # Append the character index from the generated text (t) and the character index of the character to the variable x

            # Predict the next character for the generated text.
            predictions = model.predict(x, verbose=0)[0] # Set a variable to hold a prediction; x is the input data, verbosity of 0 to suppress logging
            next_index = sample(predictions, creativity) # Get the predicted next_index value from an array of predictions using the specified level of creativity
            next_char = indices_char[next_index] # Set a next_char variable to an actual character using the inverse character indices variable

            # Update generated text. Note that we overwrite the text moving from left to right, therefore overwriting the initial seed text with all generated text.
            generated = generated[1:] + next_char # Remove first letter of seed and add next_char at the end; this is then used in the next loop to predict the next character

        # Append the generated text to the output
        outputs.append(generated)

    # Save outputs to file.
    text_file = open('outputs/' + outputFilename, "w")
    text_file.write('\n'.join(outputs))
    text_file.close()
