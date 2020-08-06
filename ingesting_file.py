import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import pandas as pd

df = pd.read_csv('bible.csv')

df_new_testament = df[df['b'] > 39]['t']

new_testament_in_verses = []
for verse in df_new_testament:
    new_testament_in_verses.append(verse)

tokenizer = Tokenizer()

tokenizer.fit_on_texts(new_testament_in_verses)
TOTAL_WORDS = len(tokenizer.word_index) + 1

input_sequences = []
for line in new_testament_in_verses:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)


print(new_testament_in_verses[0])
max_sequence_length = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences, maxlen = max_sequence_length, padding = 'pre'))

xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=TOTAL_WORDS)
batch_size = 500

model = tf.keras.models.Sequential()
ES = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=2)

model.add(tf.keras.layers.Embedding(TOTAL_WORDS, 240, input_length=max_sequence_length-1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512)))
model.add(tf.keras.layers.Dense(TOTAL_WORDS, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(xs, ys, epochs=75, batch_size=batch_size, verbose=1, callbacks=ES)

tokenizer_json = tokenizer.to_json()
with open('new_test_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

model.save('new_testament_bible_verse_model.h5')