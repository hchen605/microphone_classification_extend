import os
import sys
sys.path.append("..")
import argparse

from tensorflow import keras
import numpy as np

from ts_dataloader import *
from models.small_fcnn_att import model_fcnn

from sklearn.metrics import log_loss

from tensorflow.compat.v1 import ConfigProto, InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

classes_3 = ['C','D','M']
classes_12 = ['C1','C2','C3','C4','D1','D2','D3','D4','D5','M1','M2','M3']
genders = ['full', 'female', 'male']

parser = argparse.ArgumentParser()
parser.add_argument("--path_3", type=str, required=True, help="model path")
parser.add_argument("--path_12", type=str, required=True, help="model path")
parser.add_argument("--gender", type=int, default=0, help="full (0), female(1), male (2)")
args = parser.parse_args()

gender = genders[args.gender]

test_csv = '../data/test_{}.csv'.format(gender)
test = load_data(test_csv)
x_test, y_test_3, y_test_12 = list(zip(*test))
x_test = np.array(x_test)

cls2label_3 = {label: i for i, label in enumerate(classes_3)}
cls2label_12 = {label: i for i, label in enumerate(classes_12)}
y_test_3 = [cls2label_3[y] for y in y_test_3]
y_test_12 = [cls2label_12[y] for y in y_test_12]
num_test_data = len(y_test_3)
y_test_3 = keras.utils.to_categorical(y_test_3, num_classes=3)
y_test_12 = keras.utils.to_categorical(y_test_12, num_classes=12)

best_model_3 = model_fcnn(3, input_shape=[128, None, 1], num_filters=[24, 48, 96], wd=0)
best_model_12 = model_fcnn(12, input_shape=[128, None, 1], num_filters=[24, 48, 96], wd=0)

path_3 = 'weight/{}/3class/{}'.format(gender, args.path_3)
path_12 = 'weight/{}/12class/{}'.format(gender, args.path_12)

best_model_3.load_weights(path_3)
best_model_12.load_weights(path_12)

y_pred_3 = best_model_3.predict(x_test)
y_pred_12 = best_model_12.predict(x_test)

preds = np.zeros(y_pred_12.shape)
prob_map = np.zeros(y_pred_12.shape[1])

for i in range(y_pred_3.shape[0]):
    prob_map[0] = y_pred_3[i][0]
    prob_map[1] = y_pred_3[i][0]
    prob_map[2] = y_pred_3[i][0]
    prob_map[3] = y_pred_3[i][0]
    prob_map[4] = y_pred_3[i][1]
    prob_map[5] = y_pred_3[i][1]
    prob_map[6] = y_pred_3[i][1]
    prob_map[7] = y_pred_3[i][1]
    prob_map[8] = y_pred_3[i][1]
    prob_map[9] = y_pred_3[i][2]
    prob_map[10] = y_pred_3[i][2]
    prob_map[11] = y_pred_3[i][2]
    preds[i, :] = y_pred_12[i, :] * prob_map

y_pred = np.argmax(preds, axis=1)
y_pred_12 = np.argmax(y_pred_12, axis=1)
y_gt = np.argmax(y_test_12, axis=1)


original_acc = np.sum(y_pred_12==y_gt) / num_test_data
overall_acc = np.sum(y_pred==y_gt) / num_test_data


print("Original acc: ", "{0:.5f}".format(original_acc))
print("2-stage acc: ", "{0:.5f}".format(overall_acc))
