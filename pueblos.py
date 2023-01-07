from ctypes import sizeof
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import json

# load ascii text and covert to lowercase
filename = "municipios_ign.json"
mun_json = json.loads(open(filename, 'r', encoding='utf-8').read())
lista_nombres = []
raw_names = ""
for pueblo in mun_json:
    nombre_pueblo = pueblo["fields"]["nameunit"]
    lista_nombres.append(nombre_pueblo)
    raw_names = raw_names + "\n" + nombre_pueblo
# create mapping of unique chars to integers
raw_names = raw_names.lower()
chars = sorted(list(set(raw_names)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

n_chars = len(raw_names)
n_vocab = len(chars)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 20
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_names[i:i + seq_length]
	seq_out = raw_names[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)

# reshape X to be [samples, time steps, features]
X = np.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="200/weights-nuevo-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=200, batch_size=64, callbacks=callbacks_list)