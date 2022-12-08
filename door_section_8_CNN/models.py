from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
import settingValue as sv
import matplotlib.pyplot as plt
def MLP(train_input, train_output, batch_size, epochs):
    input_shape = Input(shape=(sv.div_data, 1))

    layer = Conv1D(8, 11, activation='relu')(input_shape)
    layer = MaxPooling1D(2)(layer)
    layer = Conv1D(16, 7, activation='relu')(layer)
    layer = MaxPooling1D(2)(layer)
    layer = Conv1D(32, 5, activation='relu')(layer)
    layer = Flatten()(layer)
    layer = Dense(64, activation='relu')(layer)
    layer = Dense(32, activation='relu')(layer)
    layer = Dense(8, activation='sigmoid')(layer)

    cnn_1d = Model(input_shape, layer)
    cnn_1d.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    hist = cnn_1d.fit(train_input, train_output,
                      batch_size=batch_size, epochs=epochs)


    plt.plot(hist.history['accuracy'])
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.savefig("accuracy graph")
    plt.show()

    return cnn_1d


def autoencoder(train_input, train_output, batch_size, epochs):
    input = Input(shape=(sv.div_data,))

    encoded = Dense(64, activation="relu")(input)
    encoded = Dense(32, activation="relu")(encoded)
    encoded = Dense(16, activation="relu")(encoded)
    encoded = Dense(32, activation="relu")(encoded)
    decoded = Dense(64, activation="relu")(encoded)
    decoded = Dense(sv.div_data, activation="sigmoid")(decoded)

    autoencoder = Model(input, decoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    autoencoder.fit(train_input, train_output,
                      batch_size=batch_size, epochs=epochs,
                      )

    return autoencoder