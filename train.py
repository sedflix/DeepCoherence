from keras.callbacks import ModelCheckpoint

from model import *
from utills import load_cui_dataset

DATASET_DIR = 'data/cui/processed/'

train, dev, test = load_cui_dataset(DATASET_DIR, MAX_SEQUENCE_LENGTH)

model = get_model()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['mae', 'acc'])
print(model.summary())

checkpoints = ModelCheckpoint('trained_models/',
                              monitor='val_loss',
                              verbose=1,
                              save_best_only=False,
                              save_weights_only=False,
                              mode='auto',
                              period=1)

model.fit(train[0], train[1],
          batch_size=128,
          epochs=10,
          shuffle=True,
          validation_data=(test[0], test[1]),
          callbacks=[checkpoints])
