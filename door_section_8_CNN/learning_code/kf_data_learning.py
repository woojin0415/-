import functions as func
import settingValue as sv
import models
import matplotlib.pyplot as plt


input_d_1 = func.pp_data(sv.steel_c_path_dir +'0.5m.csv', sv.div_data, True)
input_d_2 = func.pp_data(sv.steel_c_path_dir +'1.0m.csv', sv.div_data, True)
input_d_3 = func.pp_data(sv.steel_c_path_dir +'1.5m.csv', sv.div_data, True)
input_d_4 = func.pp_data(sv.steel_c_path_dir +'2.0m.csv', sv.div_data, True)
input_d_5 = func.pp_data(sv.steel_c_path_dir +'2.5m.csv', sv.div_data, True)
input_d_6 = func.pp_data(sv.steel_c_path_dir +'3.0m.csv', sv.div_data, True)
input_d_7 = func.pp_data(sv.steel_c_path_dir +'3.5m.csv', sv.div_data, True)
input_d_8 = func.pp_data(sv.steel_c_path_dir +'4.0m.csv', sv.div_data, True)

input_d = input_d_1 + input_d_2 + input_d_3 + input_d_4 + input_d_5 + input_d_6 + input_d_7 + input_d_8



output_d = []

for i in range(8):
    sector = [0,0,0,0,0,0,0,0]
    for j in range(int(sv.entire_data-sv.div_data)):
        sector[i] = 1
        output_d.append(sector)

####################Proximity Detector MLP####################
proximity_detector = models.MLP(input_d, output_d, 64, 150)
proximity_detector.save("kf_prox_dt.h5")
##############################################################

co_0 = func.proximity_detect_algo_0(sv.steel_c_path_dir +'0.5m_test.csv', proximity_detector, True)
co_1 = func.proximity_detect_algo_0(sv.steel_c_path_dir +'1.0m_test.csv', proximity_detector, True)
co_2 = func.proximity_detect_algo_0(sv.steel_c_path_dir +'1.5m_test.csv', proximity_detector, True)
co_3 = func.proximity_detect_algo_0(sv.steel_c_path_dir +'2.0m_test.csv', proximity_detector, True)
co_4 = func.proximity_detect_algo_0(sv.steel_c_path_dir +'2.5m_test.csv', proximity_detector, True)
co_5 = func.proximity_detect_algo_0(sv.steel_c_path_dir +'3.0m_test.csv', proximity_detector, True)
co_6 = func.proximity_detect_algo_0(sv.steel_c_path_dir +'3.5m_test.csv', proximity_detector, True)
co_7 = func.proximity_detect_algo_0(sv.steel_c_path_dir +'4.0m_test.csv', proximity_detector, True)

acc = ( co_0[0] + co_1[1] + co_2[2] + co_3[3] + co_4[4] + co_5[5] + co_6[6] + co_7[7] ) / (sv.entire_data_test - sv.div_data) / 8 * 100

print(acc)