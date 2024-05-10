from glob import glob
import numpy as np
import tensorflow as tf
from scipy.io import wavfile
from scipy.signal import stft
from sklearn.model_selection import train_test_split

datasets_dir = "TurEV-DB-master\\Sound Source\\"
input_folder_list = glob(datasets_dir + "*")

max_len = 200  # Spektrogramların boyutuna göre ayarlanmalı

audio_inputs = []
audio_targets = []
target_id = -1
for input_folder in input_folder_list:
    wav_files = glob(input_folder + "\\*")
    target_id += 1
    for wav_file in wav_files:
        sampling_freq, audio = wavfile.read(wav_file)
        _, _, Zxx = stft(audio, fs=sampling_freq, nperseg=1024)
        abs_Zxx = np.abs(Zxx)
        if abs_Zxx.shape[1] < max_len:
            pad_width = max_len - abs_Zxx.shape[1]
            abs_Zxx = np.pad(abs_Zxx, ((0, 0), (0, pad_width)), mode='constant')
        abs_Zxx = abs_Zxx[:, :max_len]  # Uzunluk sabitleme
        audio_inputs.append(abs_Zxx)
        audio_targets.append(target_id)

audio_inputs = np.array(audio_inputs)
audio_targets = np.array(audio_targets)

x_train, x_test, y_train, y_test = train_test_split(audio_inputs, audio_targets, test_size=0.25, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(x_train.shape[1], x_train.shape[2])),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(np.unique(y_train)), activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
print("Modelin doğruluğu:", model.evaluate(x_test, y_test)[1])
