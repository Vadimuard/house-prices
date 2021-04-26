import os

DATA_FOLDER = 'data'
TRAIN_CSV = os.path.join(DATA_FOLDER, 'train.csv')
VAL_CSV = os.path.join(DATA_FOLDER, 'val.csv')
TEST_CSV = os.path.join(DATA_FOLDER, 'test.csv')
SAVED_ESTIMATOR = os.path.join('models', 'GBR.pickle')
