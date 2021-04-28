import os
import json
import random
import math
import numpy as np
import time
from datetime import timedelta
import psutil


def list2string(lst, no_char_list):
    str_data = str(lst)
    for c in no_char_list:
        if c in str_data:
            str_data = str_data.replace(c, "")
    return str_data


def encode(data, value):
    str_data = list2string(data, [" ", ",", "[", "]"])
    n = 8
    res = [str_data[i:i + n] for i in range(0, len(str_data), n)]
    for i in range(len(res)):
        if len(res[i]) != n:
            res[i] = res[i] + ('0' * (n - len(res[i])))
        res[i] = chr(int(res[i], 2))
        # res[i] = res[i].encode('utf-8')
    # res = ''.join(res)
    value = str(value).replace(" ", "")
    return res, value


def decode(data, value, var_size):
    res = []
    n = 8
    for c in data:
        bits = bin(ord(c))[2:]
        if len(bits) != n:
            bits = ('0' * (n - len(bits))) + bits
        res.append(bits)
    print("res", res)
    str_data = list2string(res, [" ", ",", "[", "]", "\'"])
    print(str_data)
    # res = str_data
    res = list(map(int, str_data[0:var_size]))
    # res = str_data[0:var_size]
    # value = json.loads(value)
    print(res, value)
    return res, value


def stringtobyte(str):
    b = bytearray()
    b.extend(map(ord, str))
    return b


def bytetostring(byte):
    str = []
    for b in byte:
        str.append(chr(b))
    return str


def lsttobyte(lst):
    str_data = list2string(lst, [" ", ",", "[", "]"])
    n = 8
    res = [str_data[i:i + n] for i in range(0, len(str_data), n)]
    for i in range(len(res)):
        if len(res[i]) != n:
            res[i] = res[i] + ('0' * (n - len(res[i])))
        res[i] = chr(int(res[i], 2))
    return stringtobyte(res)


def bytetolst(byte, var_size):
    lst = bytetostring(byte)
    res = []
    n = 8
    for c in lst:
        bits = bin(ord(c))[2:]
        if len(bits) != n:
            bits = ('0' * (n - len(bits))) + bits
        res.append(bits)
    str_data = list2string(res, [" ", ",", "[", "]", "\'"])
    return list(map(int, str_data[0:var_size]))


def stringtomatrix(val, matrix_size):
    vals = val.split(",")
    lst = [vals[i:i+matrix_size] for i in range(0, len(vals), matrix_size)]
    for i in range(len(lst)):
        lst[i] = list(map(int, lst[i]))
    return lst


def add(data, value, tab_data, tab_val):
    d, v = encode(data, value)
    tab_data.append(stringtobyte(d))
    tab_val.append(json.loads(v))
    ens = zip(tab_data, tab_val)
    pairs = sorted(ens)
    return [list(tuple) for tuple in zip(*pairs)]


def dichotomy(element, sort_list):
    start, end = 0, len(sort_list)-1
    while start <= end:
        mid = (start+end)//2
        if element == sort_list[mid]:
            return mid
        elif element < sort_list[mid]:
            end = mid - 1
        else:
            start = mid + 1
    return False


def get(val, tab_data, tab_val):
    tmp = lsttobyte(val)
    # if tmp in tab_data:
    #     index = tab_data.index(tmp)
    #     return bytetolst(tab_data[index], len(val)), tab_val[index]
    index = dichotomy(tmp, tab_data)
    if index:
        return bytetolst(tab_data[index], len(val)), tab_val[index]
    else:
        return None



def pretty_val(value):
    str_ = '('
    for i in range(len(value)):
        for j in range(len(value)):
            if (i and j) != len(value) - 1:
                str_ = str_ + str(value[i][j]) + ','
            else:
                str_ = str_ + str(value[i][j]) + ')'
    return str_


def dump(tab_data, tab_val, filename):
    a = os.path.join(os.getcwd() + '/tab', filename + '.txt')
    f1 = open(a, "wb")
    for i in range(len(tab_data)):
        # data = bytetostring(tab_data[i])
        # value = pretty_val(tab_val[i])
        # for d in data:
        #     f1.write(d)
        # f1.write(value)
        tmp = '('
        for j in range(len(tab_val[i])):
            for k in range(len(tab_val[j])):
                if j == len(tab_val[i])-1 and k == len(tab_val[j])-1:
                    tmp = tmp + str(tab_val[i][j][k])
                else:
                    tmp = tmp + str(tab_val[i][j][k]) + ','
        tmp = tmp + ')'
        f1.write(tab_data[i])
        f1.write(stringtobyte(tmp))
    f1.close()


def dump_test(iter_size, size):
    a = os.path.join(os.getcwd() + '/tab', 'dump_test.txt')
    f1 = open(a, "wb")
    tab = []
    vals = []
    for i in range(iter_size):
        str1 = [chr(random.randint(0, 255)) for _ in range(int(math.ceil(size/8)))]
        val = "(" + str(random.randint(0, 10000)) + ',' + str(random.randint(0, 10000)) + ',' + \
               str(random.randint(0, 10000)) + ',' + str(random.randint(0, 10000)) + ')'
        tab.append(stringtobyte(str1))
        vals.append(stringtobyte(val))
    # print(tab)
    # print(len(tab))
    # print(vals)
    # print(len(vals))
    for i in range(len(tab)):
        # data = bytetostring(tab[i])
        # for d in data:
        f1.write(tab[i])
        f1.write(vals[i])
    f1.close()


def read(size, matrix_size, filename):
    a = os.path.join(os.getcwd() + '/tab', filename + '.txt')
    f1 = open(a, "rb")
    datas = []
    vals = []
    size = int(math.ceil(size/8))
    b = f1.read(size)
    while b:
        datas.append(bytearray(b))
        v = ''
        while b != b')':
            b = f1.read(1)
            v = v + chr(ord(b))
        vals.append(stringtomatrix(v[1:len(v)-1], matrix_size))
        b = f1.read(size)
    f1.close()
    ens = zip(datas, vals)
    pairs = sorted(ens)
    return [list(tuple) for tuple in zip(*pairs)]


def read_test(size, matrix_size):
    a = os.path.join(os.getcwd() + '/tab', 'dump_test.txt')
    f1 = open(a, "rb")
    datas = []
    vals = []
    size = int(math.ceil(size/8))
    b = f1.read(size)
    while b:
        # print(b, str(ord(b)))
        # print(b)
        datas.append(bytearray(b))
        v = ''
        while b != b')':
            b = f1.read(1)
            v = v + chr(ord(b))
        vals.append(stringtomatrix(v[1:len(v)-1], matrix_size))
        b = f1.read(size)
    f1.close()
    # print(datas)
    # print(vals)
    ens = zip(datas, vals)
    pairs = sorted(ens)
    return [list(tuple) for tuple in zip(*pairs)]


def init(size, matrix_size, filename):
    tab_data = []
    tab_val = []
    try:
        tps = time.time()
        tab_data, tab_val = read(size, matrix_size, filename)
        print("temps exe tableau init: " + str(timedelta(seconds=(time.time() - tps))))
        return tab_data, tab_val
    except:
        print("Le tableau n'existe pas encore !")
        return tab_data, tab_val


def test(itersize, size, matrix_size):
    # ECRITURE
    debut = time.time()
    dump_test(itersize, size)
    print("temps ecriture:", (timedelta(seconds=(time.time() - debut))), time.time() - debut)
    # LECTURE
    debut = time.time()
    tab_data, tab_vals = read_test(size, matrix_size)
    print("temps lecture:", (timedelta(seconds=(time.time() - debut))), time.time() - debut)
    # AJOUT
    debut = time.time()
    d1 = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
          0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1,
          0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0,
          1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1,
          0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0,
          0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1,
          0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
          0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
          0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1]
    v1 = [[1803, 173], [33, 398]]
    tab_data, tab_vals = add(d1, v1, tab_data, tab_vals)
    print("temps ajout:", (timedelta(seconds=(time.time() - debut))), time.time() - debut)
    # RECUPERATION
    debut = time.time()
    d, v = get(d1, tab_data, tab_vals)
    v = np.array(v)
    print(v)
    print(get(d1, tab_data, tab_vals))
    print("temps recuperation:", (timedelta(seconds=(time.time() - debut))), time.time() - debut)
    # AUTRE INFO
    print("nombre de solutions:", len(tab_data))
    print("taille des solutions:", size)
    print("taille des matrices:", matrix_size)
    print("mémoire utilisée:", psutil.virtual_memory())


if __name__ == '__main__':
    test(10000, 299, 2)


