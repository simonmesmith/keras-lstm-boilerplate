'''
Inspired by (and largely copied from) this:
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py

Here's my understanding of how this works:
1. Get the text
2. Create a set of unique characters from the text
3. Create two arrays of indexes of these characters; one character: index and one index: character (so we can easily translate between the two)
4. Cut up the text into bite sized strings of text ("sentences") and from this create two arrays; one of "sentences," and one of the first character that follows a sentence
5. Vectorize both the sentences array and the character that follows the sentences array
6. Create a neural network
7. Get a random string of text to seed predictions
8. Use the random string of text to generate next characters and build out novel strings, with various levels of "creativity" using a diversity setting
9. Repeat steps 7 and 8

Here are some things I'd like to do:
1. Experiment with this: Instead of generating a random seed, explicitly provide the seed or seeds to the application

'''

from __future__ import print_function
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.models import load_model
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import os.path
import numpy as np
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

# Set the maximum length of a "sentence" (used throughout the code).
maxlen = 40

# Set a variable to hold the compiled model's filename.
modelFileName = 'model.h5'

# If we already have a compiled model...
if os.path.exists(modelFileName):

    # Set a variable to hold the compiled model.
    model = load_model(modelFileName)

# Else create, compile and save the model.
else:

    # Cut the text in "semi-redundant sequences of maxlen characters," according to the author.
    step = 3 # Steps between numbers (e.g. 0, 3, 6, 9)
    sentences = [] # Variable to hold "sentences"
    next_chars = [] # Variable to hold the next character after each sentence
    for i in range(0, len(text) - maxlen, step): # Loop from 0 to the text length minus the maximum sentence length in increments defined by the step variable
        sentences.append(text[i: i + maxlen]) # Append a sentence using the index as the starting point
        next_chars.append(text[i + maxlen]) # Append the first character that comes after the sentence

    # Vectorize the sentences for analysis.
    # To learn: Who do we use the boolean type for X and y variables below?
    # To learn: Why do we set X[] and y[] to equal 1? What's the significance of that?
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool) # Create an empty (?) array for training data with the shape specified and the data type boolean
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool) # Create an empty (?) array for target data (what we want to predict) with the shape specified and the data type boolean
    for i, sentence in enumerate(sentences): # For each sentence in sentences (i = index, sentence = sentence)
        for t, char in enumerate(sentence): # For each character in the sentence (t = index, char = character)
            X[i, t, char_indices[char]] = 1 # Add to training data array (sentence index, sentence character index, index of character in char_indices array)
            y[i, char_indices[next_chars[i]]] = 1 # Add to target data array (sentence index, index of character in char_indices for the character that comes after the sentence)

    # Create neural network.
    # To learn: What are "units" in this context?
    # To learn: What does the softmax activation function do? What is "softmax?"
    model = Sequential() # Model is sequential
    model.add(LSTM(128, input_shape=(maxlen, len(chars)))) # Add an LSTM with 128 units and an input shape of maxlen (sentence maximum length) and the number of unique characters in the text (len(chars))
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
        epochs=20 # Number of times to iterate over the training data in each iteration
    )

    # Save the model with the specified file name.
    model.save(modelFileName)

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
for iteration in range(1, 2): # For iterations in the range specified...

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

        # Create a variable for the generated sentence and get a string to use as the seed.
        # To learn: Why do we use sys.stdout.write() below instead of print()?
        generated = '' # Sets variable for generated sentence
        sentence = text[start_index: start_index + maxlen] # Gets a string of maxlen length from the text using the random start index variable
        generated += sentence # Sets the generated variable to the sentence
        print('----- seed:', sentence) # Prints sentence seed
        sys.stdout.write(generated) # To learn: Why is this here versus below? What actually happens here? Gets written to buffer? Since generated set to nothing above? What happens if I remove this?

        # Loop and generate text.
        # To learn: What's the significance of doing this 400 times?
        for i in range(400):

            # Vectorize the sentence seed.
            # To learn: Why do we set x[] = 1? Why does that 1 have a period following it?
            x = np.zeros((1, maxlen, len(chars))) # Set a variable to an array of the specified shape that will hold the sentence seed in vectorized form
            for t, char in enumerate(sentence): # Loop through each character in the sentence, with t as the character index and char as the character
                x[0, t, char_indices[char]] = 1. # Append the character index from the sentence (t) and the character index of the character to the variable x

            # Predict the next character for the random sentence seed.
            # To learn: Why do we use sample(preds, diversity)? Assumption: Preds returns an array of likely next characters, with some having a higher probability
            # than others, and a lower diversity setting selects a more probable character whereas a higher diversity setting selects a less probable (i.e. more
            # creative) character. Is this true?
            preds = model.predict(x, verbose=0)[0] # Set a variable to hold a prediction; x is the input data, verbosity of 0 to suppress logging
            next_index = sample(preds, diversity) # Take a random sample of the prediction using the diversity value
            next_char = indices_char[next_index] # Set a next_char variable to an actual character using the inverse character indices variable

            generated += next_char # Add the next character to the generated variable
            sentence = sentence[1:] + next_char # Adds the next_char to the sentence? To learn: What's going on here? Ah, I think it's adding the next character to the original sentence and then having the system predict against that sentence on the next loop

            sys.stdout.write(next_char) # To learn: Is this getting buffered here? I think it may be writing each character to the screen; true?
            sys.stdout.flush() # To learn: What's getting flushed here? The buffered next_char written to sys.stdout.write() above?

        print() # To learn: Just printing a blank line here, or somehow helping get the generated text flushed?
