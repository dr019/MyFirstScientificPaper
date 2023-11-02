import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
from copy import copy, deepcopy
import math as m

import warnings
warnings.filterwarnings("ignore")

#i - по вертикали, j - по горизонтали
# по краям не генерируем
n = 200
road = 0
railway = 1
river = 2

#Функция вывода изображения
def draw(x):
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes()
    ax.set_axis_off()
    ax.imshow(x, interpolation='none',cmap='RdPu')

#Функции для загрузки началаьного поля клеточного автомата
def upload_datafield(road, river, railway):
    datafield = np.zeros((3, n, n))
    A_road = np.array(Image.open(road))
    B_river = np.array(Image.open(river))
    C_railway = np.array(Image.open(railway))
    white_road = len(A_road[0, 0])*255
    white_river = len(B_river[0, 0])*255
    white_railway = len(C_railway[0, 0])*255

    for i in range(n):
        for j in range(n):
            if sum(A_road[i, j]) != white_road:
                datafield[0, i, j] = 1
            if sum(B_river[i, j]) != white_river:
                datafield[1, i, j] = 1
            if sum(C_railway[i, j]) != white_railway:
                datafield[2, i, j] = 1
    return datafield

def init_town(town):
    x = np.zeros((n, n))
    A_town = np.array(Image.open(town))
    white = len(A_town[0, 0])*255

    for i in range(n):
        for j in range(n):
            if sum(A_town[i, j]) != white:
                x[i, j] = 1
    return x

def compress_datafield(datafield):
    new_datafield = [[], [], []]
    for k in range(3):
        for i in range(n):
            for j in range(n):
                if datafield[k][i][j] == 1:
                    new_datafield[k].append((i, j))
    return new_datafield

#Basic code
def is_road(x, i, j, datafield):
    return datafield[road][i][j] == 1

def is_railway(x, i, j, datafield):
    return datafield[railway][i][j] == 1

def is_river(x, i, j, datafield):
    return datafield[river][i][j] == 1

def applying(x, i, j, func, datafield):
    return int(func(x, i-1, j, datafield)) + int(func(x, i-1, j+1, datafield)) + \
    int(func(x, i-1, j-1, datafield)) + int(func(x, i, j+1, datafield)) + int(func(x, i, j-1, datafield)) + \
    int(func(x, i+1, j, datafield)) + int(func(x, i+1, j-1, datafield)) + int(func(x, i+1, j+1, datafield))

def sumMoor(x, i, j):
    return x[i-1, j] + x[i-1, j+1] + x[i-1, j-1] + x[i, j+1] + x[i, j-1] + \
    x[i+1, j] + x[i+1, j-1] + x[i+1, j+1]

def rules(x_prev, x_new, i, j, datafield):
    close_river = False
    close_road = False
    close_railway = False

    if i < n-1 and i > 0 and j > 0 and j < n-1:
        if x_prev[i, j] == 0:
            if sumMoor(x_prev, i, j) >= 3 and sumMoor(x_prev, i, j) <= 6:
                x_new[i, j] = 1
            elif (sumMoor(x_prev, i, j) == 2 or sumMoor(x_prev, i, j) == 1) and applying(x_prev, i, j, is_road, datafield) > 0:
                x_new[i, j] = 1
            elif (sumMoor(x_prev, i, j) == 2 or sumMoor(x_prev, i, j) == 1) and applying(x_prev, i, j, is_railway, datafield) > 0:
                x_new[i, j] = 1
            elif (sumMoor(x_prev, i, j) == 2 or sumMoor(x_prev, i, j) == 1) and applying(x_prev, i, j, is_river, datafield) > 0:
                x_new[i, j] = 1
        else:
            x_new[i, j] = 1
    else:
        x_new[i,j] = 0

def run(iters, x, datafield):
    x_prev = deepcopy(x)
    x_new = np.zeros((n, n))

    for k in range(iters):
        for i in range(n):
            for j in range(n):
                rules(x_prev, x_new, i, j, datafield)
        x_prev = deepcopy(x_new)
        x_new = np.zeros((n, n))
    return x_prev

#Upgraded code
def dist(i1, j1, i2, j2):
    a = abs(i1 - i2)
    b = abs(j1 - j2)

    return m.sqrt(a**2 + b**2)

def closest_road(i, j, compressed_datafield):
    min_dist = n
    tmp_dist = 0

    for k in range(len(compressed_datafield[0])):
        l, m = compressed_datafield[0][k]
        tmp_dist = dist(i, j, l, m)
        if (tmp_dist < min_dist):
            min_dist = tmp_dist
    
    return min_dist

def closest_river(i, j, compressed_datafield):
    min_dist = n
    tmp_dist = 0

    for k in range(len(compressed_datafield[1])):
        l, m = compressed_datafield[1][k]
        tmp_dist = dist(i, j, l, m)
        if (tmp_dist < min_dist):
            min_dist = tmp_dist
    
    return min_dist

def closest_railway(i, j, compressed_datafield):
    min_dist = n
    tmp_dist = 0

    for k in range(len(compressed_datafield[2])):
        l, m = compressed_datafield[2][k]
        tmp_dist = dist(i, j, l, m)
        if (tmp_dist < min_dist):
            min_dist = tmp_dist
    
    return min_dist



def P_g(features, W):
    res = sum([features[i]*W[i] for i in range(len(W))])

    return 1.0/(1 + m.exp(-res))


def Omega(x, N, i, j):
    cur_sum = 0

    for k in range(-N, N+1):
        for m in range(-N, N+1):
            if (0 < (i+k) < n-1) and (0 < (j+m) < n-1) and ((k != 0) or (m != 0)) and x[i+k][j+m] == 1:
                cur_sum += 1

    return float(cur_sum)/((2*N+1)**2 - 1)


def P_c(x, i, j, compressed_datafield, restricted, gamma, alpha):
    #возможно добавление запретных территорий как бинарный слой
    #gamma - стохастический коэффициент в диапазоне от 0 до 1
    #alpha - параметр регулирующий степень стохастичности

    a = closest_road(i, j, compressed_datafield)
    b = closest_river(i, j, compressed_datafield)
    c = closest_railway(i, j, compressed_datafield)
    denom = m.sqrt(2*n*n)
    P = P_g([a/denom, b/denom, c/denom], [0.5, 0.15, 0.35])

    O = Omega(x, 1, i, j)
    
    return P * O #* (1 + (-m.log(gamma))**alpha)

def main_rules(x_prev, x_new, i, j, compressed_datafield, Q, restricted, gamma, alpha):
    if i < n-1 and i > 0 and j > 0 and j < n-1:
        if x_prev[i, j] == 0:
            tmp = P_c(x_prev, i, j, compressed_datafield, restricted, gamma, alpha)
            if tmp >= Q:
                x_new[i, j] = 1
            else:
                x_new[i, j] = 0
        else:
            x_new[i, j] = 1
    else:
        x_new[i,j] = 0

def main_run(iters, x, compressed_datafield):
    x_prev = deepcopy(x)
    x_new = np.zeros((n, n))

    for k in range(iters):
        print(k)
        for i in range(n):
            for j in range(n):
                main_rules(x_prev, x_new, i, j, compressed_datafield, Q=0.20, restricted=None, gamma=3, alpha=0)
        x_prev = deepcopy(x_new)
        x_new = np.zeros((n, n))
    return x_prev