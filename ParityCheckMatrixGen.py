'''
Created on Jun 1, 2016

@author: laurynas
'''

from numpy import *

def gallager_matrix(number_of_parity_check_eqns,
                    number_of_bits,
                    column_weight,
                    row_weight):
    
#     print number_of_parity_check_eqns/row_weight
    basic_sub_matrix = zeros((number_of_bits/row_weight, number_of_bits))
    for i in range(1,number_of_bits/row_weight+1):
        for j in range((i-1)*row_weight+1, i*row_weight+1):
            basic_sub_matrix[i-1,j-1]+=1
#     print basic_sub_matrix
            
    perm_sub_matrix = basic_sub_matrix.copy()
    
    for t in range(1,column_weight):
        x = random.permutation(number_of_bits)
        perm_sub_temp = basic_sub_matrix[:,x]
        perm_sub_matrix = concatenate((perm_sub_matrix,perm_sub_temp), axis = 1)
    
    parity_check_matrix = zeros((number_of_parity_check_eqns, number_of_bits))
    for p in range(1,column_weight+1):
        parity_check_matrix[(p-1)*(number_of_bits/row_weight):(p)*(number_of_bits/row_weight),:number_of_bits] +=perm_sub_matrix[:,(p-1)*number_of_bits:p*number_of_bits] 
 
    return parity_check_matrix


if __name__ == '__main__':
    number_of_bits = 24
    column_weight = 2
    row_weight = 4
    set_printoptions(edgeitems = 100)
    number_of_parity_check_eqns = int(number_of_bits*column_weight/row_weight)
    print gallager_matrix(number_of_parity_check_eqns, number_of_bits, column_weight, row_weight)