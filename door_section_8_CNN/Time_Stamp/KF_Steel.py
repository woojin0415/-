import functions as func
import settingValue as sv
import tensorflow as tf
import time
import csv
import joblib
from sklearn import svm

iter = 100000

data_set = func.pp_data(sv.steel_c_path_dir+"\\1.0m_test.csv",sv.div_data)
time_stamp = []

data = data_set[0]
list = []
list.append(data)
data = list


###############전처리1##################
ts_1 = time.time()
for  i in range(iter):
    data1 = func.filtering2(data)
ts_2 = time.time()
time_stamp.append(ts_1)
time_stamp.append(ts_2)
###############전처리1##################
data = data1

###############전처리2##################
ts_3 = time.time()
for  i in range(iter):
    pass
ts_4 = time.time()
time_stamp.append(ts_3)
time_stamp.append(ts_4)
###############전처리2##################


###############테스트##################
test_model = tf.keras.models.load_model("steel_model_2000\\kf_prox_dt.h5")
ts_5 = time.time()
for i in range(iter):
    sector = test_model.predict(data)
ts_6 = time.time()
time_stamp.append(ts_5)
time_stamp.append(ts_6)
###############테스트##################

file = open("kf_time_steel_cnn.csv", 'w')
writer = csv.writer(file)
writer.writerow(time_stamp)
file.close()
