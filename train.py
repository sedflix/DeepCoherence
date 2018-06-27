from keras.callbacks import ModelCheckpoint

from model import *
from utills import load_cui_dataset, f1

DATASET_DIR = 'data/cui/processed/'

train, dev, test = load_cui_dataset(DATASET_DIR, MAX_SEQUENCE_LENGTH)

model = get_model()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc', f1])
print(model.summary())

checkpoints = ModelCheckpoint('trained_models/model.{epoch:02d}-{val_loss:.2f}.hdf5,',
                              monitor='val_loss',
                              verbose=1,
                              save_best_only=True,
                              save_weights_only=False,
                              mode='auto',
                              period=1)

model.fit(train[0], train[1],
          batch_size=128,
          epochs=20,
          shuffle=True,
          validation_data=(test[0], test[1]),
          callbacks=[checkpoints])
