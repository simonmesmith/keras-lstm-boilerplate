from __future__ import print_function
import helper
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import os.path
import random
import sys

def get_model(input_filename, scan_length, epochs):

    # Set a variable to hold the compiled model's filename.
    input_filename_without_extension = input_filename.split('.')[0]
    model_filename = input_filename_without_extension + '.h5'
    model_filepath = 'models/' + model_filename

    # If we already have a compiled model...
    if os.path.exists(model_filepath):

        # Set a variable to hold the compiled model.
        model = load_model(model_filepath)

        # Return the model.
        return model

    # Else create, compile and save the model.
    else:

        # Populate key variables
        text = helper.get_text(input_filename)
        unique_strings = helper.get_unique_strings(text)
        unique_string_indices = helper.get_unique_string_indices(unique_strings)
        indices_unique_string = helper.get_indices_unique_string(unique_strings)

        # Scan through the text and build an array of "if_strings" and "then_strings" that follow them, for the model to learn from.
        # For example, in the sentence "I am going for a walk," the if_string might be "I am going for a wal" and "k" would be the then_string.
        step = 3 # Steps between numbers (e.g. 0, 3, 6, 9)
        if_strings = [] # Variable to hold if_string
        then_strings = [] # Variable to hold then_string
        for i in range(0, len(text) - scan_length, step): # Loop from 0 to the text length minus the scan length in increments defined by the step variable
            if_strings.append(text[i: i + scan_length]) # Append an if_string using the index as the starting point
            then_strings.append(text[i + scan_length]) # Append a then_string that comes after the if_string

        # Vectorize the strings for analysis.
        # To learn: Why do we use the boolean type for X and y variables below? (I think for one-hot encoding.)
        # To learn: Why do we set X[] and y[] to equal 1? What's the significance of that? (I think for one-hot encoding, but I don't know for sure if this is right, nor fully understand how one-hot encoding works.)
        X = np.zeros((len(if_strings), scan_length, len(unique_strings)), dtype=np.bool) # Create an empty (?) array for training data with the shape specified and the data type boolean
        y = np.zeros((len(if_strings), len(unique_strings)), dtype=np.bool) # Create an empty (?) array for target data (what we want to predict) with the shape specified and the data type boolean
        for i, if_string in enumerate(if_strings): # For each if_string in if_strings...
            for t, char in enumerate(if_string): # For each character in the if_string (t = index, char = character)...
                X[i, t, unique_string_indices[char]] = 1 # Add to training data array (name index, chunk character index, index of character in unique_string_indices array)
                y[i, unique_string_indices[then_strings[i]]] = 1 # Add to target data array (name index, index of character in unique_string_indices for the character that comes after the chunk)

        # Create neural network.
        model = Sequential() # Model is sequential
        model.add(LSTM(128, input_shape=(scan_length, len(unique_strings)))) # Add an LSTM with 128 memory units and an input shape of scan_length (name maximum length) and the number of unique characters in the text (len(unique_strings))
        model.add(Dense(len(unique_strings))) # Add a Dense layer with as many units as there are unique characters in the text (len(unique_strings)); not that it infers the input shape from the previous layer
        model.add(Activation('softmax')) # Add the output layer with a softmax activation function (learn more at https://en.wikipedia.org/wiki/Softmax_function)

        # Set optimizer variable.
        # To learn: What do optimizers do in the context of machine learning?
        optimizer = RMSprop(lr=0.01) # RMSprop is apparently a particularly good optimizer for recurrent neural networks; lr is the learning rate, which effectively determines how fast the neural network learns from its mistakes (too fast and it will abandon its "beliefs" too often)

        # Compile model.
        model.compile(loss='categorical_crossentropy', optimizer=optimizer) # Compile with loss function (which the neural network will use to optimize against) and optimizer; to learn: why "categorical_crossentropy?"

        # Train model. Note that in the @fchollet's original version (https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py),
        # this is inside the generator loop, and it sequentially trains with each iteration. I believe the purpose of that was simply to show how the model
        # improves with each iteration, and that this isn't necessary for a more production-oriented approach.
        model.fit( # Train the model
            X, # Training data (populated above)
            y, # Target data (what to predict; populated above)
            batch_size=128, # Number of samples per gradient update
            epochs=epochs # Number of times to pass over the training data
        )

        # Save the model with the specified file name.
        model.save(model_filepath)

        # Return the  model.
        return model
