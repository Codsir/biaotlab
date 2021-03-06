#!usr\bin\python
# -*- coding:utf-8-*-

import numpy as np

#---------------------------------------
# expand ndarray:inputs must be numpy 2darray
'''
output: a 2d array wanted
'''
def expand_2d(array_A, array_B, Dimension):
    shape_A = array_A.shape
    shape_B = array_B.shape
    shouldEqualDimen = 1 - Dimension
    if shape_A[shouldEqualDimen]!=shape_B[shouldEqualDimen]:
        print 'Error!Another dimension except' \
               'what you have inputed should be equal'
    else:
        if shouldEqualDimen == 0:
            lineNumber = shape_A[0]
            columnNumber = shape_A[1]+shape_B[1]
            returnArray = np.zeros((lineNumber, columnNumber))
            returnArray[:, 0:shape_A[1]] = array_A
            returnArray[:, shape_A[1]:(shape_A[1]+shape_B[1])] = array_B
        else:
            lineNumber = shape_A[0] + shape_B[0]
            columnNumber = shape_A[1]
            returnArray = np.zeros((lineNumber, columnNumber))
            returnArray[0:shape_A[0], :] = array_A
            returnArray[shape_A[0]:(shape_A[0]+shape_B[0]), :] = array_B
    return returnArray
        
'''
find(data, value): used to find the position
'''
#--------------------------------------

'''
get the absolute value vector or matrix
'''
def bt_abs(nparray):
    return map(lambda x:abs(x), nparray)
#-------------------------------------------
'''
get the sum of ndarray
'''
def bt_sum(nparray):
    return reduce(lambda x,y:x+y, nparray)
#------------------------------------
'''
NewData= kelserConstruct(Data, class_number)
input: sample data where row is a sample
output: return a ndarray where every row
is a new sample with kelser construct

'''

def kelserConstruct(Data, class_number):
    D_shape = np.shape(Data)
    N = D_shape[0]
    d_num = D_shape[1]
    row_num =(class_number-1)*N
    col_num = class_number*d_num
    NewData = np.zeros((row_num, col_num))
    for i in range(0, N):
        for j in range(1, class_number):
            NewData[i*(class_number-1)+j-1, 0:d_num] =Data[i,:]
            NewData[i*(class_number-1)+j-1, j*d_num:(j+1)*d_num] = -Data[i,:]
    return NewData
#--------------------------------------
'''
judge if there is any value that is minus
'''

def isAnyMinus(nd_array):
    abs_array = map(lambda x:abs(x)-x, nd_array)
    sum_array = reduce(lambda x,y:x+y, abs_array)
    if sum_array:
        return  1
    else:
        return  0
    
