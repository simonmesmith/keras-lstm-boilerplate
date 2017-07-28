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

def getModel(inputFilename):

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
        maxlen = helper.getMaxLen(text)

        # Cut the text in "semi-redundant sequences of maxlen characters," according to the author.
        step = 3 # Steps between numbers (e.g. 0, 3, 6, 9)
        chunks = [] # Variable to hold chunks (chunks = strings of text)
        next_chars = [] # Variable to hold the next character after each chunk
        for i in range(0, len(text) - maxlen, step): # Loop from 0 to the text length minus the maximum chunk length in increments defined by the step variable
            chunks.append(text[i: i + maxlen]) # Append a chunk using the index as the starting point
            next_chars.append(text[i + maxlen]) # Append the first character that comes after the chunk

        # Vectorize the chunks for analysis.
        # To learn: Who do we use the boolean type for X and y variables below?
        # To learn: Why do we set X[] and y[] to equal 1? What's the significance of that?
        X = np.zeros((len(names), maxlen, len(chars)), dtype=np.bool) # Create an empty (?) array for training data with the shape specified and the data type boolean
        y = np.zeros((len(names), len(chars)), dtype=np.bool) # Create an empty (?) array for target data (what we want to predict) with the shape specified and the data type boolean
        for i, chunk in enumerate(names): # For each chunk in chunks (i = index, chunk = chunk)
            for t, char in enumerate(name): # For each character in the chunk (t = index, char = character)
                X[i, t, char_indices[char]] = 1 # Add to training data array (name index, chunk character index, index of character in char_indices array)
                y[i, char_indices[next_chars[i]]] = 1 # Add to target data array (name index, index of character in char_indices for the character that comes after the chunk)

        # Create neural network.
        # To learn: What are "units" in this context?
        # To learn: What does the softmax activation function do? What is "softmax?"
        model = Sequential() # Model is sequential
        model.add(LSTM(128, input_shape=(maxlen, len(chars)))) # Add an LSTM with 128 units and an input shape of maxlen (name maximum length) and the number of unique characters in the text (len(chars))
        model.add(Dense(len(chars))) # Add a Dense layer with as many units as there are unique characters in the text (len(chars)); not that it infers the input shape from the previous layer
        model.add(Activation('softmax')) # Add the output layer with a softmax activation function

        # Set optimizer variable.
        # To learn: What do optimizers do in the context of machine learning?
        # To learn: What is learning rate? What does this do?
        optimizer = RMSprop(lr=0.01) # RMSprop is apparently a particularly good optimizer for recurrent neural networks; lr is the learning rate

        # Compile model.
        # To learn: How do loss functions work? Why are there different ones?
        # To learn: What is the "categorical_crossentropy" loss function? Why do we use it here?
        model.compile(loss='categorical_crossentropy', optimizer=optimizer) # Compile with loss function and optimizer;

        # Train model. Note that in the author's original version, this is inside the loop below, and sequentially trained with each iteration. I believe the purpose of that
        # was simply to show how the model improves with each iteration, and that this isn't necessary for a more production-oriented approach.
        model.fit( # Train the model
            X, # Training data (populated above)
            y, # Target data (what to predict; populated above)
            batch_size=128, # Number of samples per gradient update
            epochs=25 # Number of times to iterate over the training data in each iteration
        )

        # Save the model with the specified file chunk.
        model.save(modelFilePath)

        # Return the  model.
        return model
