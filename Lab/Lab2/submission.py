## import modules here 

import numpy as np


################# Question 1 #################

# x = [3, 1, 18, 11, 13, 17]
# num_bins = 4


def v_opt_dp(x, num_bins):  # do not change the heading of the function
    global x_copy, num_bins_copy, matrix, matrix_index
    matrix = [[-1 for i in range(len(x))] for j in range(num_bins)]
    matrix_index = [[-1 for i in range(len(x))] for j in range(num_bins)]
    x_copy = x
    num_bins_copy = num_bins
    opt_dp(0, num_bins - 1)  # The bin should subtract one because the last line in matrix only has one valid solution
    result = matrix_index[-1][0]
    result_copy = result
    bins = [x[:result]]
    for i in range(len(matrix_index) - 2, 0, -1):
        result = matrix_index[i][result]
        bins.append(x[result_copy:result])
        result_copy = result
        # print(bins)
    bins.append(x[result_copy:])  # append the last bin
    # print(bins)
    return matrix, bins


def opt_dp(xmatrix, rest_of_bins):
    global x_copy, num_bins_copy, matrix, matrix_index
    if (num_bins_copy - rest_of_bins - xmatrix < 2) and (
            len(x_copy) - xmatrix > rest_of_bins):  # judge whether the entry is valid
        opt_dp(xmatrix + 1, rest_of_bins)
        if rest_of_bins == 0:
            matrix[rest_of_bins][xmatrix] = np.var(x_copy[xmatrix:]) * len(x_copy[xmatrix:])  # calculate the sse
            return
        else:
            opt_dp(xmatrix, rest_of_bins - 1)
            min_list = [matrix[rest_of_bins - 1][xmatrix + 1]]
            for i in range(xmatrix + 2, len(x_copy)):
                min_list.append(matrix[rest_of_bins - 1][i] + np.var(x_copy[xmatrix:i]) * (
                        i - xmatrix))  # use previous sse which is minimal
            matrix[rest_of_bins][xmatrix] = min(min_list)  # use minimal sse to replace the entry
            matrix_index[rest_of_bins][xmatrix] = min_list.index(min(min_list)) + xmatrix + 1
