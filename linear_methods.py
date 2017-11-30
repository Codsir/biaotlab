#!usr\bin\python
# -*- coding:utf-8-*-

#author:Tiger Zhang (Zhang Biao)
#2017-11-11

'''
this program contain several classification
methods:
Batch Perception
Ho-Kashyap algorithm
MSE(mean square error) classification algorithm

'''

import numpy as np
import biaotlab.data_methods as btdm

'''
(a_final, ite_step, error_final) = batch(Data, error_value, a_vec_0, Eta)
input: Data, every row is a sample

'''
# batch(Xdata, Ydata)
def batch(Data, error_value, a_vec_0, Eta):
   D_shape = np.shape(Data)
   data_amount = D_shape[0]
   ite_step = 0
   error_final = 100
   while error_final >= error_value:
      Y_sum = np.zeros(np.shape(a_vec_0))
      for i in range(0, data_amount):
         if a_vec_0.dot(Data[i, :].T)[0] <= 0:
            Y_sum += Data[i, :]
      a_vec_0 += Eta * Y_sum
      error_final = btdm.bt_sum(btdm.bt_sum(btdm.bt_abs(Eta * Y_sum)))
      ite_step += 1
   return (a_vec_0,ite_step, error_final )
#---------------------------------------------
'''
(a_final, error_final, ite_step) = btlm.hoKashyap(Data, a_0, b_0, error_min, K_max)

'''
def hoKashyap(Data, a_0, b_0, error_min, K_max, Eta):
   D_shape = np.shape(Data)
   data_amount = D_shape[0]
   ite_step = 0
   error_final = 100
   for i in range(0, K_max):
      e_vec = Data.dot(a_0)-b_0
      e_plus = (1/2.0) * (e_vec + btdm.bt_abs(e_vec))
      #print btdm.bt_abs(e_vec)
      b_0 = b_0 + 2*Eta*e_plus
      Y_plus = np.linalg.pinv(Data)
      a_0 = Y_plus.dot(b_0)
      abs_e =btdm.bt_sum(btdm.bt_sum(btdm.bt_abs(e_vec)))
      ite_step += 1
      if abs_e <= error_min:
         return (a_0, abs_e, ite_step)
      if i == K_max-1:
         print 'No solution found!maybe there are problem on the inputs '\
            'or you should change to another method.'
         return (a_0, abs_e, ite_step)
#-----------------------------------------------------

'''
MSE-multiclassification:Kelser
(train_error, test_accuracy, train_itestep, a_final) = mseKelser(trainData, testData, error_bound, a_vec, Eta)

'''
def mseKelser(trainData, testData, error_bound, a_vec, Eta):
   (a_final, train_itestep, train_error) = batch(trainData, error_bound, a_vec, Eta)
   value_array = map(lambda x:x.dot(a_vec.T), testData)
   test_accuracy = 0.0
   for sam in value_array:
      if sam >= 0:
         test_accuracy  += 1
   varray_shape =  np.shape(value_array)
   test_accuracy = test_accuracy/varray_shape[0]
   return (train_error, test_accuracy, train_itestep, a_final)

