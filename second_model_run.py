import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import pandas as pd

model = tf.keras.models.load_model('new_testament_bible_verse_model.h5')

df = pd.read_csv('bible.csv')

df_lev = df[df['b'] == 2]['t']

lev_in_verses = []
for verse in df_lev:
    lev_in_verses.append(verse)