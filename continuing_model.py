import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import pandas as pd
from tensorflow.keras import mixed_precision
import dask.dataframe as ddf
import numpy as np

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.experimental.set_policy(policy)

df = pd.read_csv('bible.csv')

df_entire_bible = df.t

dask_entire_bible = ddf.from_pandas(df_entire_bible, npartitions=12)

entire_bible_in_verses = []

for verse in dask_entire_bible:
    entire_bible_in_verses.append(verse)

tokenizer = Tokenizer(num_words=8192)
TOTAL_WORDS = 8192

tokenizer.fit_on_texts(entire_bible_in_verses)

input_sequences = []
for line in entire_bible_in_verses:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


max_sequence_length = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre'))

xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=TOTAL_WORDS)

new_model = tf.keras.models.load_model('JAN2021_post.h5')

new_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=2e-5), metrics=['accuracy'])

history = new_model.fit(xs, ys, batch_size=512, epochs=5, verbose=1)

new_model.save('JAN2021_pre.h5')
