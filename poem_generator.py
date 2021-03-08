import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


with open('JAN2021.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

model = tf.keras.models.load_model('JAN2021_pre.h5')

seed_text = 'father son and the holy'

next_words = 1

for _ in range(next_words):
    seed_list = tokenizer.texts_to_sequences([seed_text])[0]
    seed_list = pad_sequences([seed_list], maxlen=83, padding='pre')
    predicted = model.predict(seed_list, verbose=0)
    output_word = ''
    for word, index in tokenizer.word_index.items():
        if index == np.argmax(predicted):
            output_word = word
            break
    seed_text = seed_text + ' ' + output_word

with open('poems.txt', 'a') as f:
    f.writelines('\n'*2)
    f.writelines(seed_text)