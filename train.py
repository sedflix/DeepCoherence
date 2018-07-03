from keras.callbacks import ModelCheckpoint

from model import *
from utills import load_cui_dataset, get_class_weights

DATASET_DIR = 'data/cui/processed/'

train, dev, test = load_cui_dataset(DATASET_DIR, MAX_SEQUENCE_LENGTH)

model = get_model()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print(model.summary())

checkpoints = ModelCheckpoint('trained_models/model.{epoch:02d}-{val_loss:.3f}.hdf5',
                              monitor='acc',
                              verbose=1,
                              save_best_only=True,
                              save_weights_only=True,
                              mode='max',
                              period=1)

model.fit(train[0], train[1],
          batch_size=500,
          epochs=30,
          shuffle=True,
          class_weight=get_class_weights(train[1]),
          validation_data=(dev[0], dev[1]),
          callbacks=[checkpoints])

print(model.evaluate(x=test[0],
                     y=test[1],
                     verbose=1))

