from __future__ import print_function
import helper
import modeller
import numpy as np
import random
import sys

def write(inputFilename, outputFilename, scanLength, outputLength, creativity=0.2, epochs=25):

    # Set key variables.
    text = helper.getText(inputFilename)
    chars = helper.getChars(text)
    char_indices = helper.getCharIndices(chars)
    indices_char = helper.getIndicesChar(chars)
    model = modeller.getModel(inputFilename, scanLength, epochs)

    # Create a function that returns a prediction from an array of predictions based on the specified "temperature."
    # This allows us to return a more or less creative (i.e. more or less probable) prediction.
    # To learn: Would like to have someone walk through each step of this function and what it's doing!
    def sample(predictions, temperature=1.0):
        predictions = np.asarray(predictions).astype('float64') # Converts the variable predictions to an array of type float64
        predictions = np.log(predictions) / temperature # Converts the variable predictions to a logarithm of predictions divided by the temperature
        exp_predictions = np.exp(predictions) # Sets a variable to hold the exponential of predictions
        predictions = exp_predictions / np.sum(exp_predictions) # Sums array elements in the variable exp_predictions
        probabilities = np.random.multinomial(1, predictions, 1) # Sets a variable to sample from a multinomial distribution
        return np.argmax(probabilities) # "Returns the indices of the maximum values along an axis"; don't understand this fully

    # Create seed text and a generated variable to hold generated text.
    start_index = random.randint(0, len(text) - scanLength - 1) # Sets the start index for the seed to a random start point with sufficient length remaining for an appropriate-length seed
    seed = text[start_index: start_index + scanLength] # Sets the seed, a string of scanLength length from the text that starts from the random start index variable
    generated = '' # Sets a generated variable to hold generated text

    # Generate text for that's the output length (number of characters) specified
    for i in range(outputLength): # Create an output of outputLength length by looping that many times and adding a character each time

        # Vectorize the generated text.
        # To learn: Why do we set x[] = 1? Why does that 1 have a period following it? Assumption: one-hot encoding.
        x = np.zeros((1, scanLength, len(chars))) # Set a variable to an array of the specified shape that will hold the generated text in vectorized form
        for t, char in enumerate(seed): # Loop through each character in the seed, with t as the character index and char as the character
            x[0, t, char_indices[char]] = 1. # Append the character index from the seed (t) and the character index of the character to the variable x

        # Predict the next character for the seed text.
        predictions = model.predict(x, verbose=0)[0] # Set a variable to hold a prediction; x is the input data, verbosity of 0 to suppress logging
        next_index = sample(predictions, creativity) # Get the predicted next_index value from an array of predictions using the specified level of creativity
        next_char = indices_char[next_index] # Set a next_char variable to an actual character using the inverse character indices variable

        # Add character to generated text.
        generated = generated + next_char

        # Create the seed for the next loop.
        seed = generated[-scanLength:]

    # Save outputs to file.
    text_file = open('outputs/' + outputFilename, "w")
    text_file.write(generated)
    text_file.close()
