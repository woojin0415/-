import functions as func
import settingValue as sv
import models


train_input = func.pp_data(sv.steel_o_path_dir   + "0.5m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_o_path_dir   + "1.0m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_o_path_dir   + "1.5m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_o_path_dir   + "2.0m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_o_path_dir   + "2.5m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_o_path_dir   + "3.0m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_o_path_dir   + "3.5m.csv", sv.div_data)
train_input += func.pp_data(sv.steel_o_path_dir   + "4.0m.csv", sv.div_data)

train_output = func.pp_data(sv.steel_o_path_dir   + "0.5m.csv", sv.div_data, True)
train_output += func.pp_data(sv.steel_o_path_dir   + "1.0m.csv", sv.div_data, True)
train_output += func.pp_data(sv.steel_o_path_dir   + "1.5m.csv", sv.div_data, True)
train_output += func.pp_data(sv.steel_o_path_dir   + "2.0m.csv", sv.div_data, True)
train_output += func.pp_data(sv.steel_o_path_dir   + "2.5m.csv", sv.div_data, True)
train_output += func.pp_data(sv.steel_o_path_dir   + "3.0m.csv", sv.div_data, True)
train_output += func.pp_data(sv.steel_o_path_dir   + "3.5m.csv", sv.div_data, True)
train_output += func.pp_data(sv.steel_o_path_dir   + "4.0m.csv", sv.div_data, True)



####################Autoencoder####################
autoencoder = models.autoencoder(train_input, train_output, 64, 100)
autoencoder.save("AKF_Open.h5")
##################################################
