import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import pandas as pd
from tensorflow.keras import mixed_precision

physical_device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_device[0], True)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

df = pd.read_csv('bible.csv')

entire_bible_in_verses = [verse for verse in df.t]

tokenizer = Tokenizer(num_words=7500)
TOTAL_WORDS = 7500

tokenizer.fit_on_texts(entire_bible_in_verses)

input_sequences = [tokenizer.texts_to_sequences([line])[0][:i+1] for line in entire_bible_in_verses for i in range(1, len(tokenizer.texts_to_sequences([line])[0]))]

max_sequence_length = max([len(x) for x in input_sequences])

input_sequences_ready = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

xs = input_sequences_ready[:, :-1]
ys = tf.keras.utils.to_categorical(input_sequences_ready[:, -1], num_classes=TOTAL_WORDS)




inputs = tf.keras.Input(shape=(max_sequence_length-1),)
x = tf.keras.layers.Embedding(TOTAL_WORDS, 512, input_length=max_sequence_length-1)(inputs)
x2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
x3 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x2)
x4 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x3)
x5 = tf.keras.layers.add([x2, x3, x4])
x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x5)
x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=2))(x)
x = tf.keras.layers.ConvLSTM2D(256, (1, 1), padding='same')(x)
x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
x = tf.keras.layers.ConvLSTM2D(256, (1, 1), padding='same')(x)
x = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(x)
x = tf.keras.layers.ConvLSTM2D(256, (1, 1), padding='same')(x)
x = tf.keras.layers.Reshape((-1, 83, 256))(x)
x = tf.keras.layers.add([x, x5])
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
outputs = tf.keras.layers.Dense(TOTAL_WORDS, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

ES = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=2)

history = model.fit(xs, ys, batch_size=1048, epochs=100, verbose=1, callbacks=[ES])

tokenizer_json = tokenizer.to_json()
with open('JAN2021.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

model.save('JAN2021.h5')

