import time

from multiprocessing import Pool

import numpy as np


def power_2(x):
    return x * x


if __name__ == '__main__':
    # a = [[1, 1, 2], [2, 2, 4]]
    # b = list(a)
    # print(a, b)
    # b[0][2] = 'a'
    # print(a, b)

    # t0 = time.time()
    # with Pool(processes=4) as pool:
    #     k = pool.map(power_2, range(20000000))
    # print(time.time() - t0)
    #
    # t0 = time.time()
    # h = [power_2(x) for x in range(20000000)]
    # print(time.time() - t0)

    # k = 361
    # x = np.arange(k)
    # med = 164
    # rest = k - 164
    # rmv_list = []
    # head = 1
    # tail = len(x) - 2
    # while len(rmv_list) < rest and head <= k // 2 <= tail:
    #     if head == tail:
    #         break
    #     rmv_list.append(head)
    #     rmv_list.append(tail)
    #     head += 2
    #     tail -= 2
    # if len(rmv_list) < rest:
    #     i = 0
    #     while len(rmv_list) < rest:
    #         rmv_list.append(i)
    #         rmv_list.append(k - i - 1)
    #         i += 2
    # if len(rmv_list) > rest:
    #     rmv_list = rmv_list[:-1]
    #
    # rmv_list.sort()
    # print(len(rmv_list), len(set(rmv_list)))
    # print(rmv_list)
    # print(set(rmv_list))
    # ind_to_keep = set(range(k)) - set(rmv_list)
    # print(len(rmv_list), rest, len(ind_to_keep), len(set(range(k))), len(set(rmv_list)))

    k = 76
    x = np.arange(k)
    med = 164
    rest = med - k
    i = 0
    count = 0
    while len(x) < 164 and count < k - 1:
        val = (x[i] + x[i+1]) / 2
        x = np.insert(x, i+1, val)
        count += 1
        i += 2
    if len(x) < 164:
        while len(x) < 164:
            i = len(x) - 1
            val = (x[i-1] + x[i]) / 2
            x = np.insert(x, i + 1, val)
    print(len(x))


