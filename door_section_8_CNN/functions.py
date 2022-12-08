import csv
import settingValue as sv
import numpy as np
import kalmanfilter as kf
import matplotlib.pyplot as plt

def list_chunk(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst)-sv.div_data, 1)]


def pp_data(file, n, kalman = False):
    data = []
    rdr = csv.reader(open(file, 'r'))
    count = 0
    if kalman == True :
        for line in rdr:
            if(line[2] == ''): break
            if count == 0:
                x_esti, P = np.array([float(line[2]), 5]), 10 * np.eye(2)
                P = 100 * np.eye(2)
                count +=1
            else:
                x_esti, P = kf.kalman_filter(float(line[2]), x_esti , P)
            data.append((-1) * x_esti[0] / 100)

    else:
        for line in rdr:
            if(line[2] == ''): break
            data.append((-1) * float(line[2]) / 100)
    return list_chunk(data, n)


def get_rssi(file, kalman = False):
    rssi = []
    rdr = csv.reader(open(file, 'r'))
    if kalman == True:
        kalman = kf.KalmanFilter()
        for line in rdr:
            v = float(line[2])
            v = int(v)
            rssi.append(kalman.filtering(v))
    else:
        for line in rdr:
            rssi.apend(float(line[2]))
    return rssi



def change_one_list(list):
    return_list = []
    for set in list:
        for value in set:
            return_list.append(value)
    return return_list


def proximity_detect_algo_0(file, detector, kalman = False):
    test = pp_data(file, sv.div_data, kalman)
    sector = detector.predict(test)

    correct = [0, 0, 0, 0, 0 , 0 , 0 ,0]

    for data_set in sector:
        sector_ans = data_set.tolist().index(max(data_set))
        correct[sector_ans] += 1

    print(correct)

    return correct

def proximity_detect_algo_1(file, autoencoder, detector, kalman = False):
    test = pp_data(file, sv.div_data, kalman)
    test = np.array(test)
    p_data = autoencoder.predict(test)
    sector = detector.predict(p_data)

    correct = [0, 0, 0, 0, 0 ,0 ,0 ,0]

    for data_set in sector:
        sector_ans = data_set.tolist().index(max(data_set))
        correct[sector_ans] += 1

    print(correct)

    return correct

def proximity_detect_algo_2(file, autoencoder, autoencoder2 ,detector, kalman = False):
    test = pp_data(file, sv.div_data, kalman)
    test = np.array(test)
    p_data = autoencoder.predict(test)
    ppp_data = autoencoder2.predict(p_data)
    sector = detector.predict(ppp_data)

    correct = [0, 0, 0, 0, 0 ,0 ,0 ,0]

    for data_set in sector:
        sector_ans = data_set.tolist().index(max(data_set))
        correct[sector_ans] += 1

    print(correct)

    return correct

def proximity_detect_algo_3(file, autoencoder, detector, kalman = False):
    test = pp_data(file, sv.div_data, kalman)
    test = np.array(test)
    p_data = autoencoder.predict(test).tolist()



    p_data = filtering2(p_data)

    sector = detector.predict(p_data)

    correct = [0, 0, 0, 0, 0 ,0 ,0 ,0]

    for data_set in sector:
        sector_ans = data_set.tolist().index(max(data_set))
        correct[sector_ans] += 1

    print(correct)

    return correct

def filtering2(list):
    index = 0
    for set in list:
        x_esti, P = np.array([set[0], 5]), 10 * np.eye(2)
        P = 100 * np.eye(2)
        for i in range(len(set)):
            x_esti, P = kf.kalman_filter(set[i], x_esti , P)
            set[i] = x_esti[0]
        list[index] = set
        index += 1
    return list

def filtering3(list):
    kf_data = []
    x_esti, P = np.array([list[0], 5]), 10 * np.eye(2)
    P = 100 * np.eye(2)
    for i in range(len(list)):
        x_esti, P = kf.kalman_filter(list[i], x_esti , P)
        kf_data.append(x_esti[0])
    return kf_data






####그래프용####
def list_chunk2(lst, n):
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def pp_data2(file, n, kalman = False):
    data = []
    rdr = csv.reader(open(file, 'r'))
    count = 0
    if kalman == True :
        for line in rdr:
            if count == 0:
                x_esti, P = np.array([float(line[2]), 5]), 10 * np.eye(2)
                P = 100 * np.eye(2)
                count +=1
            else:
                x_esti, P = kf.kalman_filter(float(line[2]), x_esti , P)
            data.append((-1) * x_esti[0] / 100)

    else:
        for line in rdr:
            data.append((-1) * float(line[2]) / 100)
    return list_chunk2(data, n)
