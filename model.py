
from tensorflow.keras import layers as tl
from tensorflow.keras import backend as K
from tensorflow.keras.models import TFModel
from tensorflow.keras.callbacks import Callback


def mlp(sig_type='complex'):
    nchn = 2 if sig_type == 'complex' else 1
    inputs = tl.Input(shape=(100, 100, nchn))
    x = tl.TimeDistributed(tl.Flatten())(inputs)
    x = tl.LSTM(128, return_sequences=True)(x)
    x = tl.TimeDistributed(tl.Dense(128, activation='tanh'))(x)
    x = tl.TimeDistributed(tl.Dense(128, activation='tanh'))(x)
    # predict amplitude
    x = tl.TimeDistributed(tl.Dense(100*nchn, activation='linear'))(x)
    # predict phase
    # phase = tl.TimeDistributed(tl.Dense(100, activation='tanh'))(x)

    outputs = x

    return TFModel(inputs=inputs, outputs=outputs)
