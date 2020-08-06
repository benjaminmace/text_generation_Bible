import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

with open('new_test_tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

model = tf.keras.models.load_model('new_testament_bible_verse_model.h5')

seed_text = 'I forgive you'

next_words = 12

for _ in range(next_words):
    seed_list = tokenizer.texts_to_sequences([seed_text])[0]
    seed_list = pad_sequences([seed_list], maxlen=15, padding='pre')
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