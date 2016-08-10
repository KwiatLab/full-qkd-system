# -*- coding: utf-8 -*-
"""
Created on Jul 7 2012

@author: daniel

A refactored implementation of Slepian-Wolf code to support both binary and nonbinary SW simply with
one unified interface, rather than having separate code for both

Note that this is a prototype. It has no error handling, or any other nice thing. It is there to quickly
get good codes running, which will later run on a GPU.
"""


# Imports the necessary calculational modules
from numpy import *
from numpy.fft import *
from scipy.sparse import lil_matrix

# ENCODE:
# Given a sparse parityMatrix, the array of inputs, and their alphabet
# this uses this information to output a parity value vector, which is
# used to decode the later error sequence
def encode(parityMatrix, inputbits, alphabet):
    # The parityMatrix has the same width as inputbits,
    # so all that really needs to be done is matrix multiplication: parityMatrix * inputbits

    return (parityMatrix.dot(inputbits) % alphabet).astype(uint16)
    
# CHECK
# Checks whether the given sequence satisfies the parity checks in the given matrix
def check(parityMatrix, inputbits, alphabet, checks):
#     print "INSW",checks,encode(parityMatrix,inputbits,alphabet) 
    return (checks == encode(parityMatrix, inputbits, alphabet))


#####################################################################################################
# The class's goal is to contain the basic calculational modules of an LDPC code, such that
# the actual algorithms end up being simple one-line calls in the final code
#
class sw_math(object):
    
    # CONVOLUTE and CONVCOL
    # These are reference functions, which implement the straight-up algorithms using book definitions
    # They are here to make sure that the actual FFTs were set up correctly. The actual useful functions
    # are implemented later
    def convolute(self, a, b):
        # Create the result array (assume a and b are the same size)
        res = zeros(len(a))
        
        # Set up the array for convolution
        b = roll(flipud(b), 1)
        
        # Do the actual convolution
        for i in xrange(len(a)):
            res[i] = sum(a * roll(b, i))
            
        return res
    
    def convcol(self, mat):
        # needs to make sure that the input matrix is within bounds
        if (mat.shape[1] <= 1): return mat
        
        # Create left and right matrices, one which contains convolutions from the left, the other
        # from the right
        left = zeros((mat.shape[0], mat.shape[1] - 1))
        right = zeros((mat.shape[0], mat.shape[1] - 1))
        
        # Copy the leftmost and the rightmost
        left[:, 0] = mat[:, 0]
        right[:, -1] = mat[:, -1]
        
        # Begin the convolution loop
        for i in xrange(left.shape[1] - 1):
            left[:, i + 1] = self.convolute(left[:, i], mat[:, i + 1])
            right[:, -2 - i] = self.convolute(right[:, -1 - i], mat[:, -2 - i])
        
        # Create the result matrix
        res = zeros(mat.shape)
        
        # Fill in the two egde values
        res[:, 0] = right[:, 0]
        res[:, -1] = left[:, -1]
        
        # Fill in all the intermediate values
        for i in xrange(left.shape[1] - 1):
            res[:, i + 1] = self.convolute(left[:, i], right[:, i + 1])
        return res
    
    def parityProbabilities_conv(self, mat, syndromes):
        # Do the definition versino of probability finding
        convresult = self.convcol(mat)
        
        # Each columns in convresult is now the probability of all the other bits being each letter.
        # We now convert it to get the probability of the columns being each letter to satisfy the check
        res = roll(flipud(convresult), 1 + syndromes, 0)
    
        # Normalize the result and return 
        return self.normalizecol(res)
    
    
    #######################
    # Actually used functions begin here
    
    
    # ADDALLCOL
    # Given a matrix mat, returns a column vector, which contains the elementwise sum of all
    # columns in the matrix
    # sums of every row
    def addallcol(self, mat):
        return sum(mat, 1)
    
    
    # ADDCOL:
    # Given a matrix mat, returns a matrix where each column is the sum of all other columns, meaning
    #
    # input:
    # a b c d
    # e f g h
    # number_of_parity_check_eqns j k l
    #
    # returns
    # b+c+d a+c+d a+b+d a+b+c
    # f+g+h e+g+h e+f+h e+f+g
    # j+k+l number_of_parity_check_eqns+k+l number_of_parity_check_eqns+j+l number_of_parity_check_eqns+j+k
    
    # addcol_fast:
    # Does the sum quickly, at the expense of possible infinities messing with the answer.
    # If any values are +-infinity, this will give the wrong answer
    
    def addcol_fast(self, mat):
        # First, find the sum of all the columns
        tot = sum(mat, 1)
        
        # Create the result array
        res = zeros(mat.shape, dtype=mat.dtype)
        
        # Fill in the result array by subtracting out each column
        for i in xrange(mat.shape[1]):
            res[:, i] = tot - mat[:, i]
            
        return res
    
    # addcol_accurate:
    # Does the addition, guaranteeing that infinities from itself don't interfere
    # with the actual addition
    def addcol_accurate(self, mat):
        # needs to make sure that the input matrix is within bounds
        if (mat.shape[1] <= 1): return mat
        
        # Create left and right matrices, one which contains additions from the left, the other
        # from the right
        left = zeros((mat.shape[0], mat.shape[1] - 1), dtype=mat.dtype)
        right = zeros((mat.shape[0], mat.shape[1] - 1), dtype=mat.dtype)
        
        # Copy the leftmost and the rightmost
        left[:, 0] = mat[:, 0]
        right[:, -1] = mat[:, -1]
        
        # Begin the addition loop
        for i in xrange(left.shape[1] - 1):
            left[:, i + 1] = left[:, i] + mat[:, i + 1]
            right[:, -2 - i] = right[:, -1 - i] + mat[:, -2 - i]
        
        # Create the result matrix
        res = zeros(mat.shape, dtype=mat.dtype)
        
        # Fill in the two egde values
        res[:, 0] = right[:, 0]
        res[:, -1] = left[:, -1]
        
        # Fill in all the intermediate values
        for i in xrange(left.shape[1] - 1):
            res[:, i + 1] = left[:, i] + right[:, i + 1]
        return res
    
    # The default addcol is chosen here
    def addcol(self, mat):
        return self.addcol_accurate(mat)
    
    
    # MULALLCOL
    # Given a matrix mat, returns a column vector, which contains the elementwise product of all
    # columns in the matrix
    def mulallcol(self, mat):
        return prod(mat, 1)
    
    # MULCOL:
    # Given a matrix mat, returns a matrix where each column is the product of all other columns, meaning
    #
    # input:
    # a b c d
    # e f g h
    # number_of_parity_check_eqns j k l
    #
    # returns
    # bcd acd abd abc
    # fgh egh efh efg
    # jkl ikl ijl ijk
    
    # mulcol_fast:
    # Does multiplication very quickly, at the expense of possible infinities.
    # If any of the values in the input are 0, this will give the wrong answer
    
    def mulcol_fast(self, mat):
        # First, find the product of all the columns
        tot = prod(mat, 1)
        
        # Create the result array
        res = zeros(mat.shape, dtype=mat.dtype)
        
        # Fill the result array with the multiplication results divided by the current column
        for i in xrange(mat.shape[1]):
            res[:, i] = tot / mat[:, i]
        
        return res
    
    # mulcol_accurate:
    # Does the multiplication, guaranteeing that zeros and infinities from itself don't interfere
    # with the actual multiplication
    def mulcol_accurate(self, mat):
        # The mulcol needs to make sure that the input matrix is within bounds
        if (mat.shape[1] <= 1): return mat
        
        # Create left and right matrices, one which contains multiplications from the left, the other
        # from the right
        left = zeros((mat.shape[0], mat.shape[1] - 1), dtype=mat.dtype)
        right = zeros((mat.shape[0], mat.shape[1] - 1), dtype=mat.dtype)
        
        # Copy the leftmost and the rightmost
        left[:, 0] = mat[:, 0]
        right[:, -1] = mat[:, -1]
        
        # Begin the multiplication loop
        for i in xrange(left.shape[1] - 1):
            left[:, i + 1] = left[:, i] * mat[:, i + 1]
            right[:, -2 - i] = right[:, -1 - i] * mat[:, -2 - i]
        
        # Create the result matrix
        res = zeros(mat.shape, dtype=mat.dtype)
        
        # Fill in the two egde values
        res[:, 0] = right[:, 0]
        res[:, -1] = left[:, -1]
        
        # Fill in all the intermediate values
        for i in xrange(left.shape[1] - 1):
            res[:, i + 1] = left[:, i] * right[:, i + 1]
        return res
    
    
    # A default is accurate
    def mulcol(self, mat):
        return self.mulcol_accurate(mat)
    
    # NORMALIZECOL
    # Given mat, normalize each column independently. Makes sure there are no negative values
    def normalizecol(self, mat):
        res = array(mat.real, copy=True, dtype=float64)
        
        # Sometimes     FFT returns tiny negative value -> normalize that to 0:
        res[res < 0.00] = 0.00
        
        # Sum up all the values in each column
        colsum = sum(res, 0)
        
        if (any(colsum <= 0.0) or any(colsum >= Inf)):
            print "Failed normalizecol!"
        
        # Divide every entry by the necessary value to normalize
        res /= colsum
        # for number_of_parity_check_eqns in xrange(res.shape[0]):
        #    res[number_of_parity_check_eqns,:] /=colsum
        
        return res
    
    # NORMALIZEROW
    # Given mat, normalize each row independently. Makes sure there are no negative values
    def normalizerow(self, mat):
        res = array(mat, copy=True, dtype=float64)
        
        # Sum up all the values in each row
        rowsum = sum(res, 1)
        
        # Divide every entry by the necessary value to normalize
        res /= rowsum.reshape((len(rowsum), 1))
        
        return res
        
    
    # NORMALIZE
    # Given array, normalizes it
    def normalize(self, arr):
        # Makes sure there are no negative values, which is possible after FFT
        arr[arr < 0.000] = 0.00
        return arr / sum(arr, dtype=float64)
    
    
    # FFTCOL
    # Given a matrix mat, returns a matrix whose columns are the fft of the corresponding input columns:
    #
    # input
    # a b c
    # d e f
    # g h number_of_parity_check_eqns
    #
    # output
    # fft(a d g)  fft(b e h) fft(c f number_of_parity_check_eqns)
    def fftcol(self, mat):
        return fft(mat, axis=0)
    
    # IFFTCOL
    # Given a matrix mat, returns a matrix whose columns are the ifft of the corresponding input columns:
    #
    # input
    # a b c
    # d e f
    # g h number_of_parity_check_eqns
    #
    # output
    # ifft(a d g)  ifft(b e h) ifft(c f number_of_parity_check_eqns)
    def ifftcol(self, mat):
        return ifft(mat, axis=0)
    
    
    # P2L
    # Changes a matrix with probability values into a matrix with logs of the probabilities
    def p2l(self, mat):
        return log(mat)
    # L2P
    # Changes a matrix with columns being logs to actual probabilities
    def l2p(self, mat):
        return e ** mat
    
    
    # P2L
    # Changes a value or array with probability of 1 to a log likelihood ratio
    def p2llr(self, p1):
        return log(p1 / (1 - p1))
        
    # L2P
    # Changes a log-likelihood ratio to probability of bit being 1
    def llr2p(self, llr):
        # llr=log(p1/p0)
        # exp(llr)=p1/(1-p1)
        # exp(llr)/(1+exp(llr)) = p1
        return exp(llr) / (1 + exp(llr))        
    
    ############################################################################################
    # Using the above calculational functions, the following code calculates probabilities from
    # matrices of alphabets.
    ############################################################################################
    
    
    # PARITYPROBABILITIES
    # Given matrix with columns being probabilities of each letter in alphabet A,with width being
    # the amount of bits in parity check, and given the syndrome they satisfy, it returns a matrix 
    # where the columns are the probabilities of each bit given all the other bits.
    # In math terms,
    # Given mat M, with Mij being probability of bit j being letter number_of_parity_check_eqns, and syndrome S,
    # It returns mat R with Rij being probability of bit j being letter number_of_parity_check_eqns given S and all Mab where b!=j
    #
    # Explanation:
    # Let's say you have multiple bits of alphabet A, and you know the probability of getting each letter.
    # You also know that the correct letters satisfy a parity check, meaning that
    #  sum of all bit values modded by alphabet = parity check value.
    # The parity check value is called the syndrome.
    #
    # You want to find the probability of each bit being a letter given all the other checks
    #
    # Example:
    # Given bits a,b,c,d each of alphabet 3 (for simplicity), where a0 is propability of bit a being 0,
    # a1 is probability of being 1, and so on, the input matrix is:
    #
    # a0 b0 c0 d0
    # a1 b1 c1 d1
    # a2 b2 c2 d2
    #
    # Say we want to find the probability of each letter in a given the probabilities of b,c and d
    # What needs to be done is a convolution of b,c, and d, which will give a column vector with
    # the probability that all 3 bits combined give a value 0, 1, or 2. (Google convolution if this
    # is black magic)
    #
    # Having this probability of the addition of the bits in bcd gives you each letter, the probability
    # of a being each value is just these probabilities redistributed such that the check is satisfied:
    # Say the check says 2. That means that the probabilities are set such that
    # a_0=bcd2
    # a_1=bcd1
    # a_2=bcd0
    #
    # since 2+0=2, 1+1=2 and 2+0=2 That gives us the answer for the first column. This is repeated for all
    # columns to give
    #
    # a_0 b_0 c_0 d_0
    # a_1 b_1 c_1 d_1
    # a_2 b_2 c_2 d_2
    #
    # Naturally, this function uses more a more efficient implementation to calculate the same thing
    
    # The parity check's calculational code is:
    # roll(flipud(ifftcol(mulcol(fftcol(matrix of bit probabilities)))),1+syndrome value,0)
    
            
    def parityProbabilities_mul(self, mat, syndromes):
        # To do the convolutions, in order to find the probabilities given all the bits, we use
        # an FFT, which allows it to be done faster for all the bits at once
        convresult = self.ifftcol(self.mulcol(self.fftcol(mat)))
        
        # Each columns in convresult is now the probability of all the other bits being each letter.
        # We now convert it to get the probability of the columns being each letter to satisfy the check
        res = roll(flipud(convresult), 1 + syndromes, 0)
        
        # Normalize the result and return 
        # TODO: Interestingly, it seems like the normalize might not be necessary
        return self.normalizecol(res)
    
    
    # This does EXACTLY the same thing as the before block, except instead of multiplying
    # it does log addition
    def parityProbabilities_log(self, mat, syndromes):
        # To do the convolutions, in order to find the probabilities given all the bits, we use
        # an FFT, which allows it to be done faster for all the bits at once
        convresult = self.ifftcol(self.l2p(self.addcol(self.p2l(self.fftcol(mat)))))
        
        # Each columns in convresult is now the probability of all the other bits being each letter.
        # We now convert it to get the probability of the columns being each letter to satisfy the check
        res = roll(flipud(convresult), 1 + syndromes, 0)
        
        # Make sure that there are no fail 0s
        # res[res<=0.00]=1E-17
        # res[res>999999999]=99999
        # Normalize the result and return 
        return self.normalizecol(res)
    
    
    def parityProbabilities(self, mat, syndromes):
#         print "This is parity probabilities coming from different parities",mat
        return self.parityProbabilities_mul(mat, syndromes)
    
    # BITPROBABILITIES
    # Given a node with A possible discrete states (read: alphabet of A), with prior probability
    # estimation for each state 'prior', and with N child nodes propagating up their estimations
    # of the probability of each state through an A by N matrix mat, this calculates the probabilities
    # to propagate down back to the nodes. IE: This is basically the algorithm for belief propagation,
    # and in this case it just happens to be that the upwards propagation ends up the same as downwards
    # propagation, so it can be thought of as both an ever-expanding tree or a graph.
    
    def bitProbabilities(self, input_matrix, probability_matrix):
        # Does nultiplication of each given all the others, and multiplies in the initial value
#         print "Dimensions:\n",self.mulcol(input_matrix)*probability_matrix.reshape((len(probability_matrix),1))
#         print "Compare",input_matrix, self.mulcol(input_matrix)
#         print "This is input matrix",input_matrix
        return self.mulcol(input_matrix) * probability_matrix.reshape((len(probability_matrix), 1))
    
    # BITVALUE
    # Finds the best guess for the value of a parity check given the prior prior probability estimation,
    # and the probabilities given the child nodes. This is a hard decision, meaning that it has no
    # probabilities associated with it.
    def bitValue(self, mat, prev):
#         print "WILL BE GUEEEESING",mat
#   Used to be         return (self.mulallcol(mat)*prev).argmax()
#   but sometimes values does not converge due to multiplication by prior probability vector.
  
        return (self.mulallcol(mat)).argmax()
    
    
    # LOGBITPROBABILITIES
    # It is exactly the same as bitProbabilities, except that as an input it takes log likelihood
    def logbitProbabilities(self, mat, prev):
#         print "prob vector",self.addcol(mat)+prev.reshape((len(prev),1))
        return self.addcol(mat) + prev.reshape((len(prev), 1))
    
    # LOGBITVALUE
    # Same as bitValue, except it uses the log likelihood matrices
    def logbitValue(self, mat, prev):
#         print self.addallcol(mat)+prev
        return (self.addallcol(mat) + prev).argmax()

############################################################################################
# The sw_mathc class is a "corrected" math class, which includes a couple tiny changes to the algorithms
# above to make them better used for LDPC coding

class sw_mathc(sw_math):
    # NEVERMIND The first fix is to replace all 0s during coding with a very small non-0 values,which avoids NaNs,
    # and therefore stops error-explosions from happening
    # zeroReplace = 1E-17
    
    # NORMALIZECOL
    # Given mat, normalize each column independently. Makes sure there are no negative values
    def normalizecol(self, mat):
        zeroReplace = 1E-17
        res = array(mat.real, copy=True, dtype=float64)
#         print mat
#         print "RES", res
        # Sometimes     FFT returns tiny negative value -> normalize that to 0:
        res[res < 0.00] = zeroReplace
        
        # Sum up all the values in each column
        colsum = sum(res, 0)
#         print"Colsums", colsum
        if (any(colsum <= 0.0) or any(colsum >= Inf)):
#             self.err("Column normalization failed! Expect error explosion!")
#             self.err("Matrix:")
#             self.err(mat)
#             self.err("Column Sums:")
#             self.err(colsum)
              print "ERROR IN NORMALIZATION############################################"
#             raise Exception("Column normalization failed!") #This should NOT happen!
#             print "Column normalization failed!"
#             return res
#             colsum = 1
        
        # Divide every entry by the necessary value to normalize
        res /= colsum
        # for number_of_parity_check_eqns in xrange(res.shape[0]):
        #    res[number_of_parity_check_eqns,:] /=colsum
        
        return res
    
    def parityProbabilities_mul(self, mat, syndromes):
        # This code fixes the fringe case of a check having either no bits or one bit
        if (mat.shape[1] == 0):
            return mat
        elif (mat.shape[1] == 1):
            res = zeros(mat.shape)
            res[syndromes, 0] = 1.0
            return res        
        # To do the convolutions, in order to find the probabilities given all the bits, we use
        # an FFT, which allows it to be done faster for all the bits at once
#         print "BEFORE LOGS",self.mulcol(self.fftcol(mat))
        convresult = self.ifftcol(self.mulcol(self.fftcol(mat)))
#         print "After LOGS",convresult

        # Each columns in convresult is now the probability of all the other bits being each letter.
        # We now convert it to get the probability of the columns being each letter to satisfy the check
        res = roll(flipud(convresult), 1 + syndromes, 0)
        
        # Replace zeros with the given value
        # res[res<=0.00]=self.zeroReplace
        # res[res>999999999]=99999    
        
        # Normalize the result and return 
        # TODO: Interestingly, it seems like the normalize might not be necessary
        return self.normalizecol(res)
        
    # This does EXACTLY the same thing as the before block, except instead of multiplying
    # it does log addition
    def parityProbabilities_log(self, mat, syndromes):
        # To do the convolutions, in order to find the probabilities given all the bits, we use
        # an FFT, which allows it to be done faster for all the bits at once
        convresult = self.ifftcol(self.l2p(self.addcol(self.p2l(self.fftcol(mat)))))
        
        # Each columns in convresult is now the probability of all the other bits being each letter.
        # We now convert it to get the probability of the columns being each letter to satisfy the check
        res = roll(flipud(convresult), 1 + syndromes, 0)
        
        # Make sure that there are no fail 0s
        # res[res<=0.00]=zeroReplace
        # res[res>999999999]=99999
        # Normalize the result and return 
        return self.normalizecol(res)
    
    def err(self, s):
        print "SW:", s

############################################################################################
# The following functions create the structure necessary for values to propagate.
# They do all the movement of data

class swnb_node(sw_mathc):
    
    def __init__(self):
        # Matrix of the inputs
        self.inputMatrix = None
        # The connections that the given node has
        self.connections = []
    
    # Appends the given connection
    def addConnection(self, conn):
        self.connections.append(conn)
#         print "connections: ",self.connections
    
    # Prepare the object for actual propagation. This needs to be called before running belief propagation,
    # and after connecting all of the nodes together into the graph structure
    def prepare(self, alphabet):
#         print "node:",self
        
#         print type(self),"Number of connections",len(self.connections)
        if (len(self.connections) == 0):
            self.err("WARNING: Node not connected!")
#             print self.syndromeValue
        # elif (len(self.connections)<2):
        #    print "Warning: Node has <2 connections!"
        self.inputMatrix = ones((alphabet, len(self.connections)))
    
    # Recieve recieves the conditional probability of the given node according to the object
    def receive(self, obj, prob):
#         prob[prob<=0] =1E-14
        self.inputMatrix[:, self.connections.index(obj)] = prob
#         print "received",self.connections.index(obj),prob
    def runAlgorithm(self):
#         print "Ima here!"
        return self.bitProbabilities(self.inputMatrix, ones(self.inputMatrix.shape[0]))
    
    def propagate(self):
        # Find the resulting probability matrix
#         print "IM HEEREEEE!!!!!!!!!!!!!!!!",type(self)
        mat = self.runAlgorithm()
#         print "MAT:",mat

        # Propagate the values to each of the associated matrix's connections
#         print "@@@@@@@@@@@@@@@@@@@@@"
        for i in xrange(len(self.connections)):
            self.connections[i].receive(self, mat[:, i])
            

############################################################################################
# Create the default bit and parity check nodes for the binary and nonbinary decoder

# The 'bp-fft' nonbinary decoder is made up of the next two classes
class SW_nbBit(swnb_node):
    def __init__(self, priorProbability):
        super(SW_nbBit, self).__init__()
        self.priorProbability = priorProbability
    def runAlgorithm(self):
#         print "HEEEE"
#         print"\n\t BITS: Input M and PRIOR M: \n\n",self.inputMatrix,"\n\n",self.priorProbability
        arg = self.bitProbabilities(self.inputMatrix, self.priorProbability)
#         print "\n\t Normalization Argument:\n",arg
        return_value = self.normalizecol(arg)
#         print"\n\t After (Normalized) Argument\n",return_value 

        return return_value

    def getValue(self):
        return self.bitValue(self.inputMatrix, self.priorProbability)

class SW_nbCheck(swnb_node):
    def __init__(self, syndrome):
        super(SW_nbCheck, self).__init__()
        self.syndromeValue = syndrome
    def runAlgorithm(self):
#         print"\n\t CHECK: Input M and syndrome value M: \n\n",self.inputMatrix,"\n\n",self.syndromeValue
        arg = self.parityProbabilities(self.inputMatrix, self.syndromeValue)
#         print "RETURN OF CHECK",arg
        return arg

# The 'log-bp-fft' nonbinary decoder is made up of the next two classes
class SW_nblogBit(swnb_node):
    def __init__(self, priorProbability):
        super(SW_nblogBit, self).__init__()
        self.priorProbability = log(priorProbability)
    def runAlgorithm(self):
#         print"\n\t Input M and PRIOR M: \n\n",self.inputMatrix,"\n\n",self.priorProbability
        arg = self.logbitProbabilities(self.inputMatrix, self.priorProbability)
#         if self.getValue() == 0:
#             print "Computed probabilities:\n",arg
#         print "\n\t Normalization Argument:\n",arg
        return_value = self.normalizecol(arg)
#         print"\n\t After (Normalized) Argument\n",return_value 
#         if self.getValue() == 0:
#             print "And after normalization:\n",return_value

        return return_value
    def getValue(self):
#         print "-------------S------------"
#         print self.inputMatrix,self.priorProbability
#         print "-------------RES----------"
        r = self.logbitValue(self.inputMatrix, self.priorProbability)
#         print "Guess would be:",r
#         print "--------------------------"
        return r

class SW_nblogCheck(swnb_node):
    def __init__(self, syndrome):
        super(SW_nblogCheck, self).__init__()
        self.syndromeValue = syndrome
    def runAlgorithm(self):
        return self.parityProbabilities_log(self.inputMatrix, self.syndromeValue)


class SW_LDPC(object):
    
    # The available decoder types
    decoders = {
        "bp-fft": (SW_nbBit, SW_nbCheck),
        "log-bp-fft": (SW_nblogBit, SW_nblogCheck)
    }
    
    def __init__(self, parityMatrix, syndromes, data_probability_matrix, decoder=None, original=None, verbose=True):
#         print "Parity matrix\n",parityMatrix
        self.parityMatrix = parityMatrix
        self.syndromeValues = syndromes
#         print data_probability_matrix
        self.alphabet = data_probability_matrix.shape[0]  # The probability matrix's column size is the alphabet
#         print "Alphabet",self.alphabet
        self.correctResult = original
        self.verbose = verbose
        self.original = original
        # The number of iterations that were run on the data
        self.iteration = 0
        
        # The integrated error for the entire decoding
        self.errorIntegral = 0
        
        # Set the current guess and its associated parity checks
        self.sequenceGuess = zeros(self.parityMatrix.shape[1], dtype=int16)
        self.sequenceFailedParities = 1        
        
        
        # The failed parities for each iteration
        self.iterFailedParities = []    
        
        # Sets the correct decoder for the sequence
        self.setDecoder(decoder)

        # Create the propagating structure using the decoder
        self.prepare(data_probability_matrix)
#         print "Data prob matrix\n",data_probability_matrix
        # Set the current sequence guess and failed parities
        self.guessSequence()
        
        self.iterFailedParities.append(self.sequenceFailedParities)
        
    # Set the decoder that the SW code will use. If none given, choose reasonable one automatically
    def setDecoder(self, decoder):
        # Check the decoder type. Note that this can only be run ONCE at the object's creation
        if (isinstance(decoder, str)):
            self.bitClass = self.decoders[decoder][0]
            self.checkClass = self.decoders[decoder][1]
        elif (isinstance(decoder, tuple)):
            self.bitClass = decoder[0]
            self.checkClass = decoder[1]
        # An unknown value was passed as decoder, so choose automatically
        elif (self.alphabet > 2):
            if (self.verbose):
                print "SW_LDPC: Using nonbinary BP-FFT ('bp-fft') decoder"
            self.bitClass = self.decoders['bp-fft'][0]
            self.checkClass = self.decoders['bp-fft'][1]
        elif (self.alphabet == 2):
            if (self.verbose):
                print "SW_LDPC: Using Sum-Product ('bp') decoder"
            self.bitClass = self.decoders['bp'][0]
            self.checkClass = self.decoders['bp'][1]
        else:
            raise "SW_LDPC: The code's alphabet must be >= 2"
    
    # Set the decoding mechanism up, connect all of the bitnodes to checknodes in the correct way and prepare to decode
    def prepare(self, prior_probability_matrix):
        # Set up the necessary arrays
        number_of_bits = self.parityMatrix.shape[1]
#         print "Number of bits",number_of_bits
#         print number_of_parity_check_eqns, "---<"
        # bits have length of big number (iterate with value 40000)         
        self.bits = [None] * number_of_bits
        # checks have length of alice_sw length (length of dataset)
        number_of_parity_check_eqns = self.parityMatrix.shape[0]   
#         print "Number of parity check eqns",number_of_parity_check_eqns
      
        self.checks = [None] * number_of_parity_check_eqns
        
        # Create all of the objects
        for i in range(number_of_bits):
#             print i,"->",prior_probability_matrix[:,i]
            self.bits[i] = self.bitClass(prior_probability_matrix[:, i])
        for i in xrange(number_of_parity_check_eqns):
            self.checks[i] = self.checkClass(self.syndromeValues[i])
        
        # Now connect the nodes together according to the parity check matrix
        checkNumbers, bitNumbers = self.parityMatrix.nonzero()
        for i in xrange(len(checkNumbers)):
            self.checks[checkNumbers[i]].addConnection(self.bits[bitNumbers[i]])
            self.bits[bitNumbers[i]].addConnection(self.checks[checkNumbers[i]])
            
        # Finally, prepare the nodes to begin propagating values!
        for i in xrange(number_of_bits):
            self.bits[i].prepare(self.alphabet)
        for i in xrange(number_of_parity_check_eqns):
            self.checks[i].prepare(self.alphabet)
        
        # And now the code is ready to start propagating!


    #####################################################################
    # Running the code
    #####################################################################
    
    
    # Returns the amount of errors in number_of_total_bits that there are between mat1 and mat2
    def errors(self, mat1, mat2):
        return sum(mat1 != mat2)    
    
    # Returns the number of errors the current sequence has compared to the known 'correct' sequence
    def distanceFromCorrect(self):
        if (self.correctResult == None):
            print "Correct result is not provided so cannot estimate error of decoded string"
#             return -1
        return float(self.errors(self.correctResult, self.sequenceGuess)) / len(self.correctResult), self.errors(self.correctResult, self.sequenceGuess)
        
    # Guesses the current values and failed probabilities    
    def guessSequence(self):
        for i in xrange(len(self.bits)):
            self.sequenceGuess[i] = self.bits[i].getValue()
        print "failed for real", sum(self.sequenceGuess != self.original)
        self.sequenceFailedParities = sum(check(self.parityMatrix, self.sequenceGuess, self.alphabet, self.syndromeValues) == False)
        print "failed acc to soft", self.sequenceFailedParities
    # Propagate bit values to parity check nodes
    def propagateBits(self):
        print "Will be propagating bits"
        for i in xrange(len(self.bits)):
            self.bits[i].propagate()
    
    # Propagate check probabilities to bits
    def propagateChecks(self):
        for i in xrange(len(self.checks)):
            self.checks[i].propagate()    
    
    # Prints the current iteration's results
    def printResults(self):
        print self.iteration, "| P:", self.sequenceFailedParities,
        errorNumber = self.distanceFromCorrect()
        if (errorNumber != None):
            print "E:", errorNumber
        else:
            print
        
        
    # Iterate decoding
    def iterate(self):
        print "Calling bit propagation"
        self.propagateBits()
        print "Calling check propagation"
        self.propagateChecks()
        print "Calling sequence guess"
        self.guessSequence()
        self.iteration += 1
        self.errorIntegral += self.sequenceFailedParities
        self.iterFailedParities.append(self.sequenceFailedParities)
        if (self.verbose):
            self.printResults()
        return self.sequenceFailedParities
    
    # Run the algorithm until there is no error, or until iterations expires
    def decode(self, iterations=50, frozenFor=10):
        # Set the finishing iteration
        iterations = self.iteration + iterations
        print "Will be iterating in main SW_LDPC class"
        # Iterate until there is either 0 error, or iterations expires
        print "failed parities", self.sequenceFailedParities
        while (self.sequenceFailedParities > 0 and self.iteration < iterations):
            self.iterate()
            # Allow ending if the algorithm got stuck at a value
            if (self.iteration > frozenFor and len(set(self.iterFailedParities[-frozenFor:])) == 1):
                break
            
        return self.getGuess()
#         return self.sequenceFailedParities
    
    # Return the current best guess sequence
    def getGuess(self):
        return self.sequenceGuess.copy()

    # Return the current iteration
def getIteration(self):
        return self.iteration

