import functions as func
import settingValue as sv
import models
import matplotlib.pyplot as plt
import tensorflow as tf

###input_data 생성###
Deno = tf.keras.models.load_model("Deno.h5")

train_input = func.pp_data(sv.steel_c_path_dir   + "0.5m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_c_path_dir   + "1.0m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_c_path_dir   + "1.5m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_c_path_dir   + "2.0m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_c_path_dir   + "2.5m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_c_path_dir   + "3.0m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_c_path_dir   + "3.5m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_c_path_dir   + "4.0m.csv", sv.div_data)

train_input = Deno.predict(train_input).tolist()

data = func.list_chunk2(func.change_one_list(train_input), (sv.entire_data - sv.div_data) * sv.div_data)


data = func.filtering2(data)

for i in range(len(data)):
    data[i] = func.list_chunk2(data[i], sv.div_data)

input_d = data[0] + data[1] + data[2] + data[3] + data[4] + data[5] + data[6] + data[7]


######### input_d: Deno --> KF

output_d = []

for i in range(8):
    sector = [0,0,0,0,0,0,0,0]
    for j in range(int(sv.entire_data-sv.div_data)):
        sector[i] = 1
        output_d.append(sector)

####################Proximity Detector MLP####################
#proximity_detector = tf.keras.models.load_model("Deno_KF_prox_dt.h5")
proximity_detector = models.MLP(input_d, output_d, 64, 150)
proximity_detector.save("Deno_KF_prox_dt.h5")
##############################################################

co_0 = func.proximity_detect_algo_3(sv.steel_c_path_dir +'0.5m_test.csv', Deno, proximity_detector)
co_1 = func.proximity_detect_algo_3(sv.steel_c_path_dir +'1.0m_test.csv', Deno, proximity_detector)
co_2 = func.proximity_detect_algo_3(sv.steel_c_path_dir +'1.5m_test.csv', Deno, proximity_detector)
co_3 = func.proximity_detect_algo_3(sv.steel_c_path_dir +'2.0m_test.csv', Deno, proximity_detector)
co_4 = func.proximity_detect_algo_3(sv.steel_c_path_dir +'2.5m_test.csv', Deno, proximity_detector)
co_5 = func.proximity_detect_algo_3(sv.steel_c_path_dir +'3.0m_test.csv', Deno, proximity_detector)
co_6 = func.proximity_detect_algo_3(sv.steel_c_path_dir +'3.5m_test.csv', Deno, proximity_detector)
co_7 = func.proximity_detect_algo_3(sv.steel_c_path_dir +'4.0m_test.csv', Deno, proximity_detector)

acc = ( co_0[0] + co_1[1] + co_2[2] + co_3[3] + co_4[4] + co_5[5] + co_6[6] + co_7[7] ) / (sv.entire_data_test - sv.div_data) / 8 * 100

print(acc)