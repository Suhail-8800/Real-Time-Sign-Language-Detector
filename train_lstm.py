import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ================= LOAD DATA =================
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")
labels = np.load("label_classes.npy")

num_classes = len(labels)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("Number of classes:", num_classes)

# ================= ONE-HOT ENCODING =================
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# ================= MODEL =================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(30, X_train.shape[2])),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= TRAIN =================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop]
)

# ================= SAVE MODEL =================
model.save("sign_language_lstm.h5")

print("\n✅ LSTM training completed and model saved.")
