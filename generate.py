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

# Here we vectorize the sentences for analysis. Let's dig into this further. I'm not
# quite sure what's happening here, but it appears that we're starting with by
# creating empty arrays of arrays. After setting the X and y variables, they
# contain what appear to be empty arrays of arrays. Then we loop through each
# sentence and have the index (i) of the sentence and the sentence itself as variables.
# Then, within each sentence, we iterate through the characters of the sentence
# and set the index (t) of the characters and the characters themselves (char) as
# variables. Then, for the last bit there, where we set X[] and y[] to 1, I don't
# understand what's going on here! To learn: What's going on with the X and y variables?
# What does numpy.zeros do?
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1


# Here we create our neural network. Yay! We're creating a sequential model. This is
# a linear stack of layers. The first layer needs the input shape and subsequent layers
# can infer it automatically. For LSTM, we specify units (128) and the input shape, which
# is a function of the sentence maximum length (maxlen) and the number of unique characters
# in the specified text. We then add a Dense layer with units set to the number of unique
# characters in the specified text. We then add an activation layer for the output and
# use a softmax activation function.
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
#
# optimizer = RMSprop(lr=0.01)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer)
#
#
# def sample(preds, temperature=1.0):
#     # helper function to sample an index from a probability array
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return np.argmax(probas)
#
# # train the model, output generated text after each iteration
# for iteration in range(1, 60):
#     print()
#     print('-' * 50)
#     print('Iteration', iteration)
#     model.fit(X, y,
#               batch_size=128,
#               epochs=1)
#
#     start_index = random.randint(0, len(text) - maxlen - 1)
#
#     for diversity in [0.2, 0.5, 1.0, 1.2]:
#         print()
#         print('----- diversity:', diversity)
#
#         generated = ''
#         sentence = text[start_index: start_index + maxlen]
#         generated += sentence
#         print('----- Generating with seed: "' + sentence + '"')
#         sys.stdout.write(generated)
#
#         for i in range(400):
#             x = np.zeros((1, maxlen, len(chars)))
#             for t, char in enumerate(sentence):
#                 x[0, t, char_indices[char]] = 1.
#
#             preds = model.predict(x, verbose=0)[0]
#             next_index = sample(preds, diversity)
#             next_char = indices_char[next_index]
#
#             generated += next_char
#             sentence = sentence[1:] + next_char
#
#             sys.stdout.write(next_char)
#             sys.stdout.flush()
#         print()
