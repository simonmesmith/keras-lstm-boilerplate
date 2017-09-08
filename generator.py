from __future__ import print_function
import helper
import modeller
import numpy as np
import random
import sys

def write(input_filename, output_filename, level='character', scan_length=10, output_length=500, creativity=0.2, epochs=25):

    # Set key variables.
    text = helper.get_text(input_filename)
    unique_strings = helper.get_unique_strings(text, level)
    unique_string_indices = helper.get_unique_string_indices(unique_strings)
    indices_unique_string = helper.get_indices_unique_string(unique_strings)
    model = modeller.get_model(input_filename, level, scan_length, epochs)

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

    # Create seed text and a generated_string variable to hold the generated string.
    strings_in_order = helper.split_strings_as_needed(text, level) # The strings to use in their original order, either an array of words or just the straight text
    random_start_index = random.randint(0, len(strings_in_order) - scan_length - 1) # Sets the start index for the seed to a random start point with sufficient length remaining for an appropriate-length seed
    seed = strings_in_order[random_start_index: random_start_index + scan_length] # Sets the seed, a string of scan_length length from the text that starts from the random start index variable
    generated_string = [] if level == 'word' else '' # Sets a generated_string variable to hold generated_string text

    # Generate text that's the output length (number of strings) specified
    for i in range(output_length): # Create an output of output_length length by looping that many times and adding a string each time

        # Vectorize the generated_string text.
        # To learn: Why do we set x[] = 1? Why does that 1 have a period following it? Assumption: one-hot encoding.
        x = np.zeros((1, scan_length, len(unique_strings))) # Set a variable to an array of the specified shape that will hold the generated_string text in vectorized form
        for t, substring in enumerate(seed): # Loop through each substring in the string, with t as the substring index
            x[0, t, unique_string_indices[substring]] = 1. # Append the substring index from the seed (t) and the string index of the substring to the variable x

        # Predict the next string.
        predictions = model.predict(x, verbose=0)[0] # Set a variable to hold a prediction; x is the input data, verbosity of 0 to suppress logging
        next_index = sample(predictions, creativity) # Get the predicted next_index value from an array of predictions using the specified level of creativity
        next_string = indices_unique_string[next_index] # Set a next_string variable to an actual string using the inverse string indices variable

        # Add string to generated_string text.
        if level is 'word':
            generated_string.append(next_string)
        else:
            generated_string = generated_string + next_string

        # Create the seed for the next loop.
        seed = generated_string[-scan_length:]

    # Save outputs to file.
    output_text = ' '.join(generated_string) if level == 'word' else generated_string
    text_file = open('outputs/' + output_filename, "w")
    text_file.write(output_text)
    text_file.close()
