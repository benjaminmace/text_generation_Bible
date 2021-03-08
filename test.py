import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import pandas as pd

df = pd.read_csv('bible.csv')

df_first = df[df['b'] > 33]['t']

df_second = df[df['b'] <= 33]['t']

df_entire_bible = df.t

first_in_verses = []
for verse in df_first:
    first_in_verses.append(verse)

second_in_verses = []
for verse in df_second:
    second_in_verses.append(verse)

entire_bible_in_verses = []
for verse in df_entire_bible:
    entire_bible_in_verses.append(verse)

tokenizer = Tokenizer()

tokenizer.fit_on_texts(entire_bible_in_verses)
TOTAL_WORDS = len(tokenizer.word_index) + 1

input_sequences = []
for line in entire_bible_in_verses:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_length = max([len(x) for x in input_sequences])
print(max_sequence_length)