"""reshape the data from the FRED database"""
import numpy as np
import math
import csv
from typing import List


def ma(lst: List[float], x: int) -> List[float]:
    """return the moving average with range x
    Preconditions:
     - len(lst) > x
    """
    res = []
    for i in range(len(lst) - x):
        temp = lst[i:i + x]
        res.append(sum(temp) / x)
    starting = [0.0] * x
    starting.extend(res)
    assert len(starting) == len(lst)
    return starting


def macd(lst: List[float]) -> List[tuple]:
    """return the macd of the list.
    return format: (dif, dem, bar value)

    Preconditions:
     - len(lst) > 26
    """
    res = []
    ma12 = ma(lst, 12)
    ma26 = ma(lst, 26)
    dif = [0.0] * 26
    for i in range(len(lst) - 26):
        dif.append(ma12[i + 26] - ma26[i + 26])
    dem = ma(dif, 9)
    for i in range(len(dif)):
        a = dif[i]
        b = dem[i]
        c = a - b
        temp = (a, b, c)
        res.append(temp)
    return res


def boll(lst: List[float]) -> List[tuple]:
    """return the bollinger band 20 of the list
    return format: (upper, mid, lower)
    """
    mid = ma(lst, 20)
    upper = [0.0] * 20
    lower = [0.0] * 20
    bb = []
    for i in range(len(lst) - 20):
        sd_temp = sd(lst[i:i + 20], mid[i:i + 20], 20)
        upper.append(sd_temp + mid[i + 20])
        lower.append(-sd_temp + mid[i + 20])
    for j in range(len(mid)):
        temp_tuple = (upper[j], mid[j], lower[j])
        bb.append(temp_tuple)
    return bb


def sd(lst: List[float], mean: List[float], x) -> float:
    """return the standard deviation of the selected list"""
    sd_sum = 0.0
    for i in range(x):
        temp = (lst[i] - mean[i]) ** 2
        sd_sum += temp
    return math.sqrt(sd_sum / x)


def read_data(file_path: str = "CBETHUSD.csv") -> List[float]:
    """read the data from file_path"""
    raw = []
    with open(file_path) as data:
        reader = csv.reader(data)
        next(reader)

        for row in reader:
            temp = float(row[1])
            raw.append(temp)
    return raw


def read_v2_file(file_path: str = "CBETHUSD.csv") -> tuple:
    """read the data from the csv file and convert it into the a list matrices"""
    raw_data = read_data(file_path)
    ma9 = ma(raw_data, 9)
    ma26 = ma(raw_data, 26)
    all_macd = macd(raw_data)
    all_boll = boll(raw_data)
    converted = []
    result = []
    for i in range(len(raw_data) - 33):  # 33 = 26 + 2 + 5
        pre = i + 26
        temp = []
        cur = pre + 1
        nex = cur + 1
        standard_mean = ma26[cur]

        day_range = raw_data[cur] - raw_data[pre]
        day_mean = raw_data[pre] + (day_range / 2)
        day_range_norm = day_range / raw_data[pre]  # 1
        temp.append(round(day_range_norm, 7))

        boll_drv = all_boll[nex][1] - all_boll[pre][1]
        boll_drv_norm = boll_drv / all_boll[pre][1]  # 2

        boll_range = (all_boll[cur][0] - all_boll[cur][2]) / 2
        boll_pos = (day_mean - all_boll[cur][1]) / boll_range  # 3
        temp.extend([round(boll_drv_norm, 7), round(boll_pos, 7)])

        macd_value = all_macd[cur][2]
        macd_value_norm = macd_value / standard_mean  # 4
        dif_drv = all_macd[nex][0] - all_macd[pre][0]
        dem_drv = all_macd[nex][1] - all_macd[pre][1]
        macd_drv = 0
        if (dif_drv > 0 and dem_drv > 0) \
                or (dif_drv < 0 and dem_drv < 0):
            macd_drv = (dif_drv + dem_drv) / standard_mean  # 5
        macd_pos = (all_macd[cur][0] + all_macd[cur][1]) / standard_mean  # 6
        temp.extend([round(macd_value_norm, 7), round(macd_drv, 7), round(macd_pos, 7)])

        ma9_drv = ma9[nex] - ma9[pre]
        ma9_drv_norm = ma9_drv / ma9[cur]  # 7
        ma9_pos = day_mean - ma9[cur]
        ma9_pos_norm = ma9_pos / day_mean  # 8
        temp.extend([round(ma9_drv_norm, 7), round(ma9_pos_norm, 7)])

        converted.append(temp)

        res = [0, 0, 0]
        if abs(raw_data[nex + 5] - raw_data[cur]) < abs(raw_data[cur] * 0.05):
            res[1] = 1
        elif raw_data[nex + 5] - raw_data[cur] < 0:
            res[0] = 1
        else:
            res[2] = 1
        # result format: (down, stay ,up)
        result.append(res)

    return (np.array(converted), np.array(result))


"""
----------------------------END------------------------------
"""


def read_v1_file(path: str = "CBETHUSD.csv") -> tuple:
    """
    Read the data from the file path, reconstruct the format the the data
    and return a 3d matrix.
    """
    lst = []
    res = []
    with open(path) as data:
        reader = csv.reader(data)
        next(reader)  # skip the header row
        for row in reader:
            lst.append(float(row[1]))

    lst_con = []
    for i in range(len(lst) - 30):
        temp = lst[i:i + 25]
        lst_con.append(temp)

        res_temp = lst[i + 30] - temp[-1]
        res_cat = [0, 0, 0]
        if abs(res_temp) < abs(temp[-1] * 0.05):
            res_cat[1] = 1
        elif res_temp < 0:
            res_cat[0] = 1
        else:
            res_cat[2] = 1
        res.append(res_cat)

    np_lst = np.array(lst_con).reshape(len(lst_con), 25, 1)
    np_res = np.array(res)
    return (np_lst, np_res)

