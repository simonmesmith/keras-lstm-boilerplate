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

def getModel(inputFilename, scanLength, epochs):

    # Set a variable to hold the compiled model's filename.
    inputFileNameWithoutExtension = inputFilename.split('.')[0]
    modelFileName = inputFileNameWithoutExtension + '.h5'
    modelFilePath = 'models/' + modelFileName

    # If we already have a compiled model...
    if os.path.exists(modelFilePath):

        # Set a variable to hold the compiled model.
        model = load_model(modelFilePath)

        # Return the model.
        return model

    # Else create, compile and save the model.
    else:

        # Populate key variables
        text = helper.getText(inputFilename)
        chars = helper.getChars(text)
        char_indices = helper.getCharIndices(chars)
        indices_char = helper.getIndicesChar(chars)

        # Scan through the text and build an array of chunks (strings of text) and characters that follow them, for the model to learn from.
        # For example, in the sentence "I am going for a walk," the chunk might be "I am going for a wal" and "k" would be the next character.
        step = 3 # Steps between numbers (e.g. 0, 3, 6, 9)
        chunks = [] # Variable to hold chunks (chunks = strings of text)
        next_chars = [] # Variable to hold the next character after each chunk
        for i in range(0, len(text) - scanLength, step): # Loop from 0 to the text length minus the maximum chunk length in increments defined by the step variable
            chunks.append(text[i: i + scanLength]) # Append a chunk using the index as the starting point
            next_chars.append(text[i + scanLength]) # Append the first character that comes after the chunk

        # Vectorize the chunks for analysis.
        # To learn: Why do we use the boolean type for X and y variables below? (I think for one-hot encoding.)
        # To learn: Why do we set X[] and y[] to equal 1? What's the significance of that? (I think for one-hot encoding, but I don't know for sure if this is right, nor fully understand how one-hot encoding works.)
        X = np.zeros((len(chunks), scanLength, len(chars)), dtype=np.bool) # Create an empty (?) array for training data with the shape specified and the data type boolean
        y = np.zeros((len(chunks), len(chars)), dtype=np.bool) # Create an empty (?) array for target data (what we want to predict) with the shape specified and the data type boolean
        for i, chunk in enumerate(chunks): # For each chunk in chunks (i = index, chunk = chunk)...
            for t, char in enumerate(chunk): # For each character in the chunk (t = index, char = character)...
                X[i, t, char_indices[char]] = 1 # Add to training data array (name index, chunk character index, index of character in char_indices array)
                y[i, char_indices[next_chars[i]]] = 1 # Add to target data array (name index, index of character in char_indices for the character that comes after the chunk)

        # Create neural network.
        model = Sequential() # Model is sequential
        model.add(LSTM(128, input_shape=(scanLength, len(chars)))) # Add an LSTM with 128 memory units and an input shape of scanLength (name maximum length) and the number of unique characters in the text (len(chars))
        model.add(Dense(len(chars))) # Add a Dense layer with as many units as there are unique characters in the text (len(chars)); not that it infers the input shape from the previous layer
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
        model.save(modelFilePath)

        # Return the  model.
        return model
