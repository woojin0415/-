import functions as func
import settingValue as sv
import models
import matplotlib.pyplot as plt
import tensorflow as tf

AKF = tf.keras.models.load_model("AKF.h5")
AKF_OPEN = tf.keras.models.load_model("AKF_Open.h5")

train_input = func.pp_data(sv.steel_c_path_dir   + "0.5m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_c_path_dir   + "1.0m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_c_path_dir   + "1.5m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_c_path_dir   + "2.0m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_c_path_dir   + "2.5m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_c_path_dir   + "3.0m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_c_path_dir   + "3.5m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_c_path_dir   + "4.0m.csv", sv.div_data)

train_input = AKF.predict(train_input)


train_output = func.pp_data(sv.steel_o_path_dir   + "0.5m.csv", sv.div_data)
train_output += func.pp_data(sv.steel_o_path_dir   + "1.0m.csv", sv.div_data)
train_output += func.pp_data(sv.steel_o_path_dir   + "1.5m.csv", sv.div_data)
train_output += func.pp_data(sv.steel_o_path_dir   + "2.0m.csv", sv.div_data)
train_output += func.pp_data(sv.steel_o_path_dir   + "2.5m.csv", sv.div_data)
train_output += func.pp_data(sv.steel_o_path_dir   + "3.0m.csv", sv.div_data)
train_output += func.pp_data(sv.steel_o_path_dir   + "3.5m.csv", sv.div_data)
train_output += func.pp_data(sv.steel_o_path_dir   + "4.0m.csv", sv.div_data)

train_output = AKF_OPEN.predict(train_output)

plt.plot(train_input.tolist()+train_output.tolist())
plt.show()

####################Autoencoder####################
autoencoder = tf.keras.models.load_model("AKF_Deno.h5")
##################################################




input_d = autoencoder.predict(train_input).tolist()

output_d = []

for i in range(8):
    sector = [0,0,0,0,0,0,0,0]
    for j in range(int(sv.entire_data-sv.div_data)):
        sector[i] = 1
        output_d.append(sector)

####################Proximity Detector MLP####################
#proximity_detector = tf.keras.models.load_model("AKF_Deno_prox_dt.h5")
proximity_detector = models.MLP(input_d, output_d, 64, 150)
proximity_detector.save("AKF_Deno_prox_dt.h5")
##############################################################

co_0 = func.proximity_detect_algo_2(sv.steel_c_path_dir +'0.5m_test.csv', AKF, autoencoder, proximity_detector)
co_1 = func.proximity_detect_algo_2(sv.steel_c_path_dir +'1.0m_test.csv', AKF, autoencoder, proximity_detector)
co_2 = func.proximity_detect_algo_2(sv.steel_c_path_dir +'1.5m_test.csv', AKF, autoencoder, proximity_detector)
co_3 = func.proximity_detect_algo_2(sv.steel_c_path_dir +'2.0m_test.csv', AKF, autoencoder, proximity_detector)
co_4 = func.proximity_detect_algo_2(sv.steel_c_path_dir +'2.5m_test.csv', AKF, autoencoder, proximity_detector)
co_5 = func.proximity_detect_algo_2(sv.steel_c_path_dir +'3.0m_test.csv', AKF, autoencoder, proximity_detector)
co_6 = func.proximity_detect_algo_2(sv.steel_c_path_dir +'3.5m_test.csv', AKF, autoencoder, proximity_detector)
co_7 = func.proximity_detect_algo_2(sv.steel_c_path_dir +'4.0m_test.csv', AKF, autoencoder, proximity_detector)


acc = ( co_0[0] + co_1[1] + co_2[2] + co_3[3] + co_4[4] + co_5[5] + co_6[6] + co_7[7] ) / (sv.entire_data_test - sv.div_data) / 8 * 100

print(acc)