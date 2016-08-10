# -*- coding: utf-8 -*-
"""
Created on Jun 28 2012

@author: daniel

All preparation of values happens here, including loading things from files and streaming data
"""

#from pylab import *
from numpy import zeros,dot,array,ceil,float64,uint16,Inf
from scipy.sparse import *
from scipy import *
import random as randmodule
from random import *
from scipy.weave import *
from matplotlib.pyplot import *
#RANDOMDATA
#Creates a random sequence of length 'amt' with an alphabet of 'alph'
def randomData(amt, alph):
    return randint(0,alph,amt)

#ERRORDATA:
#Adds error to matrix 'mat' with alphabet 'alph' such that it has an error rate 'err' (in percentage),
def errorData(mat,alph,err):
    errorLocs = invert((random(len(mat))/err).astype(int).astype(bool))
    errorValues = randint(1,alph,len(mat))
    return (mat+errorLocs*errorValues)%alph


#TRANSITIONMATRIX_SYMMETRIC
#Given an error percentage, assuming a symmetric channel, creates the associated transition matrix
def transitionMatrix_symmetric(alph,err):
    res = ones((alph,alph),dtype=float64)*(err/(alph-1))
    res += diag(ones(alph)*(1-err)-(err/(alph-1)))
    return res
    

#SEQUENCEPROBMATRIX
#Given a sequence and a transition matrix, this returns a matrix where each column vector is the
# corresponding letter's transition probability to each alphabet letter
def sequenceProbMatrix(seq,trans):
    res = zeros((trans.shape[0],len(seq)))
    for el in xrange(len(seq)):
        # takes every column and assigns transmat columns
#         print seq[el]          
        res[:,el]=trans[:,int(seq[el])]
  
    return res

#############################################################################################################
#Analysis of code and other helper functions


#NORMALIZECOL
#Given mat, normalize each column independently. Makes sure there are no negative values
def normalizecol(mat):
    res = array(mat.real,copy=True,dtype=float64)
    
    #Sometimes     FFT returns tiny negative value -> normalize that to 0:
    
    res[res<0.00]=0.00
    
    #Sum up all the values in each column
    colsum= sum(res,0)
    # Do I want to check zero?     
    if (any(colsum<=0.0) or any(colsum >= Inf)):
#         print "Failed normalizecol:",colsum
        colsum[colsum<=0.0]=1.0
    
    #Divide every entry by the necessary value to normalize
    res/=colsum
    #for number_of_parity_check_eqns in xrange(res.shape[0]):
    #    res[number_of_parity_check_eqns,:] /=colsum
    
    return res


#ERRORS
#Returns the amount of errors in total that there are between mat1 and mat2
def errors(mat1,mat2):
    return sum(mat1!=mat2)

#ERRORPERCENT
#Returns the error rate
def errorPercent(mat1,mat2):
    return errors(mat1,mat2)/float(len(mat1))

#TRANSITIONNUMBERS_DATA
#Returns the non-probability equivalent of a transition matrix, with actual numbers,
#rather than their probabilities
#Meaning: letters from mat1 are columns and mat2 are rows
def transitionNumbers_data(mat1,mat2,alph):
    # overloads memory     
    lmat2 = zeros((len(mat1),alph))
    lmat1 = zeros((alph,len(mat2)))
    
    for number_of_parity_check_eqns in xrange(alph):
        lmat1[number_of_parity_check_eqns,:][mat2==number_of_parity_check_eqns] = 1
        lmat2[:,number_of_parity_check_eqns][mat1==number_of_parity_check_eqns] = 1
    return dot(lmat1,lmat2)
def transitionNumbers_data2_python(mat1,mat2,alph):
    rmat=zeros((alph,alph))
    datalen=int(len(mat1))
    alph=int(alph)
    i = 0
    while i <datalen:
#         print mat1[i]," - ",mat2[i]
#         print (mat2[i])
        rmat[int(mat2[i])][int(mat1[i])]+=1
        i+=1
#     print rmat

#    trying to fix 0 probability problem
    for i in range(alph):
        if sum(rmat[:,i]) == 0: 
            rmat[i][i] +=1
#         print "Col",i,sum(rmat[:,i])
    for i in range(alph):
        for j in range(alph):
            if rmat[i][j] == 0:
                rmat[i][j] = 1E-17
    return rmat
    
def transitionNumbers_data2(mat1,mat2,alph):
    rmat=zeros((alph+1,alph+1))
    datalen=int(len(mat1))
    alph=int(alph+1)
    code="""
        long long i = 0;
        for (;i<datalen;i++) {
            rmat[mat2[i]*alph+mat1[i]]+=1;
        }
    """
    inline(code,['mat1','mat2','rmat','datalen','alph'])
    return rmat
#TRANSITIONMATRIX_DATA
#Returns a transition matrix from mat1 to mat2
#This means that given an alphabet value from mat1, to get the probabilities of mat2 
#we just matrix-multiply a column vector of size alphabet made of all 0s with a 1 in the letter's location
#Meaning
#Say of alphabet 5, we choose a value 3. this means that to get the probability of each letter 
#in mat2 given this, we do Matrix product of:
# TransitionMatrix * [0 0 0 1 0] = probability
#
def transitionMatrix_data(mat1,mat2,alph):
    return normalizecol(transitionNumbers_data(mat1,mat2,alph))
def transitionMatrix_data2(mat1,mat2,alph):
    return normalizecol(transitionNumbers_data2(mat1,mat2,alph))
def transitionMatrix_data2_python(mat1,mat2,alph):
    return normalizecol(transitionNumbers_data2_python(mat1,mat2,alph))    
#SHOWMAT
#Shows a visual representation of the matrix by colorcoding values
def showmat(mat):
    imshow(mat,interpolation='nearest')
    colorbar()
    show()



#############################################################################################################
#Maniputation of parity check matrix

#WRITEMATRIX
#Writes the sparse matrix to file. The input matrix will be converted to lil_matrix format
def writeMatrix(mat,fname):
    mat=mat.asformat("lil")
    f = open(fname,"w")
    f.write(str(mat.shape[0])+" "+str(mat.shape[1])+" 0\n")
    for number_of_parity_check_eqns in xrange(len(mat.rows)):
        for j in xrange(len(mat.rows[number_of_parity_check_eqns])):
            f.write(str(number_of_parity_check_eqns)+" " + str(mat.rows[number_of_parity_check_eqns][j]) + " " + str(mat.data[number_of_parity_check_eqns][j])+ "\n")
    f.close()

#READMATRIX
#Reads a sparse matrix file, and returns the corresponding lil_matrix
def readMatrix(fname):
    f=loadtxt(fname,dtype=int64)
    return coo_matrix((f[1:,2],(f[1:,0],f[1:,1])),shape=(f[0,0],f[0,1]),dtype=uint16).asformat("csr")


#Creates a random matrix with 'checks' parity checks and 'bits' total bits
#  checks - number of parity check eqns
#  bits - number of bits
#  parities - 1s in parity check matrix of simply number of edges between bit nodes->parity nodes
#  this is how many parities does eahc bit have (column weight)
def randomMatrix(bits,checks,parities=3):
    #The matrix is created using its transpose to add parity checks to each bit
    parity_check_matrix = lil_matrix((bits,checks),dtype = uint16)
#     print parity_check_matrix
    for bit_value in xrange(parity_check_matrix.shape[0]):
        j=0
        while (j<parities):
            loc = randmodule.randrange(parity_check_matrix.shape[1])
            if (parity_check_matrix[bit_value,loc]==0):
                parity_check_matrix[bit_value,loc]=1
                j+=1
    return parity_check_matrix.transpose()

#Makes sure each row has a minimum of "min" entries
def rowmin(mat,rentries):
    failrows=0
    mat = mat.asformat("lil").copy()
#     print mat, "printing rowmin mat"
    for number_of_parity_check_eqns in xrange(len(mat.rows)):
        pass
#         print len(mat.rows[number_of_parity_check_eqns]),"<",rentries+1
#WTF?Infinite loop
#         while (len(mat.rows[number_of_parity_check_eqns])<rentries+1):    #The +1 is there due to identity
#             loc = randmodule.randrange(0,mat.shape[1])
#             if (mat[number_of_parity_check_eqns,loc]==0):
#                 mat[number_of_parity_check_eqns,loc]=1
#                 failrows+=1
    #print "Added",failrows,"entries"
    return mat

#Make sure each column has a minimum number of entries
def colmin(mat,entries):
    return rowmin(mat.transpose(),entries).transpose()

def normrow(mat):
    #Normalizes each row to have the same number of entries
    mat = mat.asformat("lil").copy()
    
    #Find the number of total elements in the matrix
    totalElementsPerCheck = len(mat.nonzero()[0])/float(mat.shape[0])
    checkBitNumberUp = ceil(totalElementsPerCheck)
    checkBitNumberDown = int(totalElementsPerCheck)
    
    #Rows which are both above and below the "tot" amount
    underRows = []
    overRows = []
    
    #Fill the underRows and overRows arrays
    for number_of_parity_check_eqns in xrange(len(mat.rows)):
        if (len(mat.rows[number_of_parity_check_eqns])>checkBitNumberUp):
            overRows.append(number_of_parity_check_eqns)
        elif (len(mat.rows[number_of_parity_check_eqns])<checkBitNumberDown):
            underRows.append(number_of_parity_check_eqns) 
    
    #print "O:",len(overRows),"U:",len(underRows),"T:",totalElementsPerCheck    
    
    #Now move elements from overRows to underRows
    for number_of_parity_check_eqns in overRows:
        if (len(underRows)<=0):
            break
        while (len(mat.rows[number_of_parity_check_eqns])>checkBitNumberUp):
            chosenValue=randmodule.choice(mat.rows[number_of_parity_check_eqns])
            chosenCheck = randmodule.randint(0,len(underRows)-1)
            
            if (mat[underRows[chosenCheck],chosenValue]==0):
                
                #Remove the value from th current row
                mat[number_of_parity_check_eqns,chosenValue]=0
                
                #Put it in the new row
                mat[underRows[chosenCheck],chosenValue] = 1
                
                #Check if 'chosenCheck' still has too-little values
                if (len(mat.rows[underRows[chosenCheck]])>=checkBitNumberUp):
                    #If it is full, remove it from the underlist
                    underRows.pop(chosenCheck)
            
                if (len(underRows)<=0):
                    break
    
    return mat

def girth(mat):
    #First check for length 4
    pass

def removeloops(mat):
    pass


#Crossover of rows in a matrix
def crossover_asym(mat1,mat2,percent):
    #Chooses len*percent checks from matrix1, and replaces them with len*percent checks from matrix2
    #This allows one parent to front most of the "DNA"
    
    #Creates the child matrix
    child = mat1.copy()
    
    #Chooses unique rows to cross-over from both parents    
    m1rows = []
    m2rows = []
    
    #Choose random rows
    for number_of_parity_check_eqns in xrange(mat1.shape[0]):
        if (randmodule.random() <= percent):
            m1rows.append(number_of_parity_check_eqns)
    for number_of_parity_check_eqns in xrange(mat2.shape[0]):
        if (randmodule.random() <= percent):
            m2rows.append(number_of_parity_check_eqns)
    
    #Normalize the rows to have the same amount
    while (len(m1rows) > len(m2rows)):
        m1rows.pop(randmodule.randint(0,len(m1rows)-1))
    while (len(m1rows) < len(m2rows)):
        m2rows.pop(randmodule.randint(0,len(m2rows)-1))
    
    #Exchange the rows!
    for number_of_parity_check_eqns in xrange(len(m1rows)):
        child[m1rows[number_of_parity_check_eqns],:]=mat2[m2rows[number_of_parity_check_eqns],:]
    
    return child
    
#Direct crossover of rows in a matrix (matrices must be same size)
def crossover(mat1,mat2,percent):
    #This is a symmetric crossover, meaning that it crosses over the same parity checks
    #The reason for this is that related matrices would have been owned by related rows
    #crossing over, which would completely destroy the advantage of crossing over
    
    #Chooses len*percent checks from matrix1, and replaces them with len*percent checks from matrix2
    #This allows one parent to front most of the "DNA"
    #It assumes the parents both have the same number of parity checks
    
    #Creates the child matrix
    child = mat1.asformat("lil").copy()
    mat2=mat2.asformat("lil").copy()
    #Chooses unique rows to cross-over from both parents    
    rows = []
    
    #Choose random rows
    for number_of_parity_check_eqns in xrange(mat1.shape[0]):
        if (randmodule.random() <= percent):
            rows.append(number_of_parity_check_eqns)
    
    #Crossover!
    for number_of_parity_check_eqns in xrange(len(rows)):
        child.rows[number_of_parity_check_eqns]=mat2.rows[number_of_parity_check_eqns]
        child.data[number_of_parity_check_eqns]=mat2.data[number_of_parity_check_eqns]
        #child[rows[number_of_parity_check_eqns],:]=mat2[rows[number_of_parity_check_eqns],:]#VERY SLOOOW
    return child

#Adds 'number' ones to the matrix
def addones(mat,number):
    mat=mat.copy()
    while (number > 0 ):
        number-=1
        mat[randmodule.randint(0,mat.shape[0]-1),randmodule.randint(0,mat.shape[1]-1)]=1
    return mat

#Deletes 'number' ones from the matrix
def delones(mat,number):
    mat=mat.copy()
    if (mat.sum <= number): return mat
    while (number > 0):
        rownum = randmodule.randint(0,mat.shape[0]-1)
        if (len(mat.rows[rownum])!=0):
            number-=1
            pnum=randmodule.choice(mat.rows[rownum])
            mat[rownum,pnum]=0
    return mat
