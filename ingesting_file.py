import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import pandas as pd

df = pd.read_csv('bible.csv')

df_new_testament = df[df['b'] > 39]['t']

df_old_testament = df[df['b'] <= 39]['t']

df_entire_bible = df.t

new_testament_in_verses = []
for verse in df_new_testament:
    new_testament_in_verses.append(verse)

old_testament_in_verses = []
for verse in df_old_testament:
    old_testament_in_verses.append(verse)

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

input_sequences = np.array(pad_sequences(input_sequences, maxlen=86, padding='pre'))

xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]
ys = tf.keras.utils.to_categorical(labels, num_classes=TOTAL_WORDS)
batch_size = 300

model = tf.keras.models.Sequential()
ES = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=3)

model.add(tf.keras.layers.Embedding(TOTAL_WORDS, 240, input_length=max_sequence_length-1))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512)))
model.add(tf.keras.layers.Dense(TOTAL_WORDS, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(xs, ys, epochs=2, batch_size=batch_size, verbose=1, callbacks=[ES])

tokenizer_json = tokenizer.to_json()
with open('entire_bible_tokens_new_testament_verses_tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

model.save('entire_bible_tokens_new_testament_verses_model.h5')