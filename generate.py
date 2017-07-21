'''
Inspired by (and largely copied from) this:
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys

# Create a variable to hold the the text. Note that we set the text to lowercase.
# To learn: Why? Does a mix of upper and lowercase mess up the neural network?
# Assumption: We list all characters in the text below. If it has uppercase and
# lowercase, that would be additional characters (for example, "a" and "A").
text = open('real-drug-brand-names.txt').read().lower()

# Set a variable to hold the characters from the text, in list format, in ascending
# order. It seems set() extracts unique characters from the text, list() turns them
# into a list() and sorted turns them into an alphabetically ordered list of characters.
chars = sorted(list(set(text)))

# This next section creates variables that hold the index of each character in the chars
# variable. For example, if chars were ['a', 'b', 'c'] then char_indices would be
# {'a': 0, 'b': 1 and 'c': 2 } while indices_char would be { 0: 'a', 1: 'b' and 2: 'c'}.
# I'm not sure yet how these are used, but it seems that this creates a key that we can
# use to translate between numbers (the indexes) and characters.

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Now we cut the text in "semi-redundant sequences of maxlen characters," according to
# the author. What's going on here? Below, the range() function takes three paramaters:
# start (the index to start at), stop (the index to stop at) and step (the steps between
# numbers). For example, range (0, 10, 2) would proceed 0, 2, 4, 6, 8, 10. Accordingly,
# the code here is saying to loop from 0 to the length of the text minus the maxlen
# variable in increments defined by the step variable. For each step, it appends appends
# values to the sentences and next_chars arrays. For sentences, it's appending a length
# of text from i to the maxlen (basically, a sentence or string of text the length of
# maxlen). For next_chars, it's appending only single characters, the first character
# after the sentence. This is a bit confusing in Python because [i: i + maxlen] is
# getting everything from i up to but not including i + maxlen. Whereas i + maxlen
# is getting that last bit. For example, [0: 0 + 10] wouldn't get 10, whereas [10]
# would. Anyway, I don't quite understand yet why we're adding these items to arrays
# in this way, but since this is a character level LSTM, it makes sense that we're
# getting characters.
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

# Next we vectorize the sentences for analysis.

# First we use numpy.zeros to create arrays with the shape
# (length/number of all sentences, max length of any one sentence,
# length/number of unique characters). For example, if we had 1000 sentences that were
# no more than 40 characters each, and 26 unique characters, we'd have a shape of
# (1000, 40, 26). The data type for these arrays are boolean, meaning true or false.
# To learn: Why are we using a boolean type here? What's the significance of that?
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

# Next we loop through each sentence, and then loop through each character in each
# sentence. As we loop, "i" is the index for each "sentence," "t" is the index of
# each character in each sentence (I think), and "char" is the character itself.
# To learn: What are we doing here with setting the X and y variables to populate What
# we created above? Why are we setting these to 1 (possibly meaning "true")?
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1 # Note that X contains the index of the sentence, each character in the sentence, and the index of the character in our character index array
        y[i, char_indices[next_chars[i]]] = 1 # Note that y contains the index of the sentence, and the index of the character that comes after the sentence, as added above
        # So it seems that what we're doing here is populating a variable with actual character sequences (X) and a variable with next characters (y) so the model can predict the latter from the former!

# Here we create our neural network. Yay! We're creating a sequential model. This is
# a linear stack of layers. The first layer needs the input shape and subsequent layers
# can infer it automatically. For LSTM, we specify units (128) and the input shape, which
# is a function of the sentence maximum length (maxlen) and the number of unique characters
# in the specified text. We then add a Dense layer with units set to the number of unique
# characters in the specified text. We then add an activation layer for the output and
# use a softmax activation function.
# To learn: What are units in this context? What does "softmax" do?
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

# Next we set a variable to hold an optimizer. I don't fully understand what optimizers do
# in the context of machine learning, but apparently RMSprop is a particularly good one for
# recurrent neural networks. To learn: What do optimizers do in machine learning?
optimizer = RMSprop(lr=0.01)

# Next we compile our model. We set the loss function and the optimizer. Loss refers to
# what our model is trying to minimize (from what I understand). The optimizer was addressed
# above. Why "categorical_crossentropy?" I'm not sure, nor quite sure what it does, but according
# to this Stack Exchange thread "Your choices of activation='softmax' in the last layer and compile
# choice of loss='categorical_crossentropy' are good for a model to predict multiple mutually-exclusive
# classes." So it seems that the loss function and the activation layer function (is that what "softmax"
# is, a function?) work together. To learn: What is categorical_crossentropy? Why is the choice of this
# important? What other variables should I consider when setting it?
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# This is a function that the author says "samples an index from a probability array." I have
# no idea what this is for yet.
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64') # Converts the variable preds to an array of type float64
    preds = np.log(preds) / temperature # Converts the variable preds to a logarithm of preds divided by the temperature
    exp_preds = np.exp(preds) # Sets a variable to hold the exponential of preds
    preds = exp_preds / np.sum(exp_preds) # Sums array elements in the variable exp_preds
    probas = np.random.multinomial(1, preds, 1) # Sets a variable to samples from a multinomial distribution
    return np.argmax(probas) # "Returns the indices of the maximum values along an axis"

# Train the model and output generated text after each iteration.
for iteration in range(1, 60):

    print() # Prints blank
    print('-' * 50) # Prints a row of hyphens
    print('Iteration', iteration) # Prints "Iteration X"

    # Trains the model. To learn or check: Why do we need to train the model for each iteration, rather
    # then training it once and then using it? I should try doing that instead, no?
    # model.fit( # Trains the model for a fixed number of epochs (iterations on a dataset).
    #     X, # Array of training data.
    #     y, # Array of target data (what to predict).
    #     batch_size=128, # Number of samples per gradient update.
    #     epochs=1 # Number of times to iterate over the training data arrays.
    # )

    # Here we set a start_index variable to be a random integer between 0 and the length of the text minus the maximum
    # length of a sentence minus 1 (inclusive). I assume the reason for this range is that we don't want to specify
    # a starting index that would result in the length of a sentence plus a character taking us beyond the range of
    # the text. To learn: Where do we use the start index?
    start_index = random.randint(0, len(text) - maxlen - 1)

    # Here we loop through diversity values. No idea why we're doing this yet. To learn: Why are we doing this?
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print() # Prints blank
        print('----- diversity:', diversity) # Prints the diversity

        generated = '' # Sets variable for generated sentence
        sentence = text[start_index: start_index + maxlen] # Gets a random string of maxlen length from the text from the random starting point
        generated += sentence # Sets the generated variable to the sentence
        print('----- Generating with seed: "' + sentence + '"')
        print(generated) # This was sys.stdout.write, but from what I've read that's the same as print() so just using print() instead

        # Looping again, from 0 to 399.
        for i in range(400):

            # Setting variable x (lowercase this time) to an array of shape (1, max length of a sentence, number of unique characters in the text).
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence): # Loop through each character in the sentence generated above
                x[0, t, char_indices[char]] = 1. # Set the x value to the character index in the sentence (t) and the index of the character in the character set (to learn: why set to equal 1?)

            preds = model.predict(x, verbose=0)[0] # Set a variable to hold a prediction; x is the input data, verbosity of 0 to suppress logging
            print ('preds')
            print(preds)
            next_index = sample(preds, diversity) # To learn: What's going on here? We're using the sample function to pluck something using the diversity variable, but why?
            print ('next_index')
            print(next_index)
            next_char = indices_char[next_index] # Setting a next_char variable to a character variable using the inverse character indices
            print('next_char')
            print(next_char)

            generated += next_char # Add the next character to the generated variable
            print('generated')
            print(generated)
            sentence = sentence[1:] + next_char # Adds the next_char to the sentence?
            print ('sentence')
            print (sentence)

            sys.stdout.write(next_char) # Write the next character
            sys.stdout.flush() # Force python to write standard out to the terminal
            print() # Print a blank row
