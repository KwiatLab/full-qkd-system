# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 2012
"""
#from pylab import *
from Statistics import *
import numpy
def log2(x):
    if x > 0:
        return numpy.log2(x)
    return 0

"""
Given a sequence with an alphabet size 'alphabet' with each letter equally likely (symmetric), the entropy per letter is:
1) entropy_per_letter =  -log_2(1/alphabet)

Given probability of having a coincidence in 2 sequences as 'p1g1', then the probability of the second sequence
having the same letter is p1g1.
The probability of the second sequence having a specific different letter is pdifferent = (1-p1g1)/(alphabet-1).

This gives the entropy of sequence 2 given sequence 1 as:
2) entropy_s2g1 =   -p1g1*log2(p1g1)-(alphabet-1)*pdifferent*log2(pdifferent)

Subtracting the two gives the maximum possible entropy left over after running a nonbinary SW code on this sequence
"""

def theoretical_nb(p1g1,alphabet):
    #p1g1 is coincidence
    #p0g1 is failcoincidence
    print "pCoincidence:",p1g1
    print "Alphabet:",alphabet
    
    #Each letter of the alphabet has the same probability
    p0g1=(1-p1g1)/(alphabet-1)
    p_letter = 1/float(alphabet)
    
    entropy_per_letter = -log2(p_letter)
    print "\nEntropy Per Letter:",entropy_per_letter
    
    entropy_s2gs1 = -p1g1*log2(p1g1)-(alphabet-1)*p0g1*log2(p0g1)
    entropy_left = entropy_per_letter-entropy_s2gs1
    
    print "Minimum Entropy Sent:",entropy_s2gs1
    print "Maximum Entropy Left:",entropy_left
    
    return entropy_left

"""
Given a sequence with an alphabet size 'alphabet' with each letter equally likely (symmetric), the entropy per letter is:
1) entropy_per_letter =  -log_2(1/alphabet)

Given probability of having a coincidence in 2 sequences as 'p1g1', then the probability of the second sequence
having the same letter is p1g1.
The probability of the second sequence having a specific different letter is pdifferent = (1-p1g1)/(alphabet-1).

This gives the entropy of sequence 2 given sequence 1 as:
2) entropy_s2g1 =   -p1g1*log2(p1g1)-(alphabet-1)*pdifferent*log2(pdifferent)

Subtracting the two gives the maximum possible entropy left over after running a nonbinary SW code on this sequence
"""


def theoretical_entropy_transition_matrix(p_letter,transmat,alphabet):
    #p1g1 is coincidence
    #p0g1 is failcoincidence
    # print "Transition Matrix:",transmat
    # print "Alphabet:",alphabet
    
    #Each letter of the alphabet has the same probability
    #p_letter = 1/float(alphabet)
    
    p_letA = numpy.dot(p_letter,transmat.transpose())
#     print transmat
#     print "letTTER PROBS:",p_letA,p_letter
#     print"---------->>"     
#     print("PA:",p_letA)
#     print("PB:",p_letter)
    entropy_per_letter = 0.0
    for number_of_parity_check_eqns in xrange(len(p_letA)):
        if (p_letA[number_of_parity_check_eqns] > 0):
            entropy_per_letter-=p_letA[number_of_parity_check_eqns]*log2(p_letA[number_of_parity_check_eqns])
    print "\nEntropy Per Letter:",entropy_per_letter
    
    entropy_s2gs1 = 0.0
    # Takes first COLUMN length     
    for number_of_parity_check_eqns in xrange(len(transmat[:,0])):
        # Takes first ROW length       
        for j in xrange(len(transmat[0,:])):
            if (transmat[number_of_parity_check_eqns,j] > 0.0 and p_letter[j] > 0.0):
                entropy_s2gs1 -= p_letter[j]*transmat[number_of_parity_check_eqns,j]*log2(transmat[number_of_parity_check_eqns,j])
    #entropy_s2gs1/=float(len(transmat[0,:]))
    
    entropy_left = entropy_per_letter-entropy_s2gs1
    
    if (entropy_left<=0.0):
        entropy_left =0.0

    print "Minimum Entropy Sent:",entropy_s2gs1
    print "Maximum Entropy Left:",entropy_left
    
    return entropy_left

"""
This calculates the same thing as the above function - except for a binary code

Given p1 as the probability of a 1 in a binary sequence
p0 = 1-p1

1) entropy_per_bit = -p1*log2(p1)-p0*log2(p0)

Given p1g1 as the probability of 1 in another sequence given this sequence (assuming that both sequences have the same statistics):
p0g1 = 1-p1g1
p1g0=p0g1*p1/p0
p0g0=1-p1g0

Then the entropy of the unknown sequence given a 1 in this sequence is
entropy_g1=-p1g1*log2(p1g1)-p0g1*log2(p0g1)

And entropy of unknown sequence given a 0 in this sequence is
entropy_g0=-p1g0*log2(p1g0)-p0g0*log2(p0g0)

Giving a total entropy per bit in the unknown sequence given this sequence:
2) entropy_s2gs1 = entropy_g1*p1+entropy_g0*p0

Subtracting the two gives the maximum possible entropy left over after running a binary SW code on this sequence

"""
def theoretical(p1a,p1g1a,p1b,p1g1b):
    
    #Probability calculations
    p0a=1-p1a
    p0b=1-p1b
    p0g1a = 1-p1g1a
    p0g1b = 1-p1g1b
    
    p1g0a=p0g1a*p1a/p0a
    p1g0b=p0g1b*p1b/p0b
    p0g0a=1-p1g0a
    p0g0b=1-p1g0b
    
#     print "A: p0   %f\tp1   %f\np0g1 %f\tp1g1 %f\np0g0 %f\tp1g0 %f\n"%(p0a,p1a,p0g1a,p1g1a,p0g0a,p1g0a)
#     print "B: p0   %f\tp1   %f\np0g1 %f\tp1g1 %f\np0g0 %f\tp1g0 %f\n"%(p0b,p1b,p0g1b,p1g1b,p0g0b,p1g0b)
    
    #Entropy calculations
    entropy_per_bit_a = -p1a*log2(p1a)-p0a*log2(p0a)
    entropy_per_bit_b = -p1b*log2(p1b)-p0b*log2(p0b)
    
    print "A: Entropy Per Bit:",entropy_per_bit_a
#     print "B: Entropy Per Bit:",entropy_per_bit_b
    
    entropy_g1a=-p1g1a*log2(p1g1a)-p0g1a*log2(p0g1a)
    entropy_g0a=-p1g0a*log2(p1g0a)-p0g0a*log2(p0g0a)
    
    entropy_agb = entropy_g1a*p1a+entropy_g0a*p0a
    
    entropy_left = entropy_per_bit_a-entropy_agb
    if (entropy_left<=0.0):
        entropy_left =0.0

    print "Minimum Entropy Sent:",entropy_agb
    print "Maximum Entropy Left:",entropy_left
    
    if (numpy.isnan(entropy_left)): return 0.0
    return entropy_left



def entropy_calculate(

    probability_of_one_in_a=0.05,        #The probability of a photon in a time bin (p1)
    coincidence_rate_a=0.4,              #The data's heralding efficiency (coincidence rate) (p1 given 1)
    probability_of_one_in_b=0.05,        #The probability of a photon in a time bin (p1)
    coincidence_rate_b=0.4,              #The data's heralding efficiency (coincidence rate) (p1 given 1)
    alphabet_size=16,                    #The alphabet (frame size used)
    b_probability_of_one=0.05,           #Probability of 1 in binary code
    b_coincidence_rate_a=0.85,           #Coincidence rate of binary slepian-wolf 1s
    b_probability_of_one_in_b=0.05,      #Probability of 1 in binary code
    b_coincidence_rate_b=0.85,           #Coincidence rate of binary slepian-wolf 1s
    coincidence_rate_non_binary=0.7,     #Probability of coincidence in nonbinary code
    nb_bperf = 64,                       #Effective number of original sequence bits per nonbinary letter
    b_test = 1.0,                        #Percentage sent in parities
    nb_test = 0.8,                       #Percentage sent in parities nonbinary
    nbpl =None,
    transition_matrix_non_binary=None    #Transition matrix for nonbinary code
    ):

    print "\n\nTHEORY:\n\n"

    print "TOTAL SEQUENCE THEORETICAL ENTROPY:"
    ideal_entropy = theoretical(probability_of_one_in_a,coincidence_rate_a,probability_of_one_in_b,coincidence_rate_b)

    print "\n\nSLEPIAN-WOLF THEORETICAL (APPROX) ENTROPY"
    #Note that the entropy retained is per alphabet bits of the original sequence
    b_entropy = theoretical(b_probability_of_one,b_coincidence_rate_a,b_probability_of_one_in_b,b_coincidence_rate_b)/alphabet_size

    print "\nPercentage of total entropy recovered:",b_entropy/ideal_entropy

    print "\n\nNON-BINARY THEORETICAL (APPROX) ENTROPY"
    #Note that the entropy retained is for a large amount of original sequence bits
    nb_entropy = 0.0
    
#     if (transition_matrix_non_binary!=None):
    nb_entropy = theoretical_entropy_transition_matrix(nbpl,transition_matrix_non_binary,alphabet_size)/nb_bperf
#     else:
#     nb_entropy = theoretical_nb(coincidence_rate_non_binary,alphabet_size)/nb_bperf
    print "NON BINARy-->>>>",nb_entropy
    print "\nPercentage of total entropy recovered:",nb_entropy/ideal_entropy

    #These are calculations of entropy retained for the actual codes
    test_b_entropy = (1-b_test)/alphabet_size
    test_nb_entropy = log2(alphabet_size)*(1-nb_test)/nb_bperf

    print "\n\nCURRENT TEST RESULTS:"
    print "              Entropy   P_theory P_total"
    print "Slepian-Wolf: %f  %f %f"%(test_b_entropy,test_b_entropy/b_entropy,test_b_entropy/ideal_entropy)
    print "Non-Binary:   %f  %f %f"%(test_nb_entropy,test_nb_entropy/nb_entropy,test_nb_entropy/ideal_entropy)
        
    print "\n\nTOTAL PERCENTAGE OF ENTROPY RETAINED:"
    print "Theory: %f"%((nb_entropy+b_entropy)/ideal_entropy)
    print "Test:   %f"%((test_nb_entropy+test_b_entropy)/ideal_entropy)

    print "Test/Theory",(test_nb_entropy+test_b_entropy)/(nb_entropy+b_entropy)
    
    print "\n\n"
    
    return (ideal_entropy,b_entropy,nb_entropy)

def entropy_calculate2(
    probability_of_one_in_a=0.05,       #The probability of a photon in a time bin (p1)
    coincidence_rate_a=0.4,             #The data's heralding efficiency (coincidence rate) (p1 given 1)
    probability_of_one_in_b=0.05,       #The probability of a photon in a time bin (p1)
    coincidence_rate_b=0.4,             #The data's heralding efficiency (coincidence rate) (p1 given 1)
    alph=16,                            #The alphabet (frame size used)
    binary_letter_probabilites=None,    #Binary Probability of each letter
    b_mat=None,                         #Transition matrix for binary code
    nb_bperf = 64,                      #Effective number of original sequence bits per nonbinary letter
    nonbinary_letter_probabilities=None,#nonbinary probability of each letter
    transition_matrix_non_binary=None   #Transition matrix for nonbinary code
    ):
    
    print "\n\nTHEORY:\n\n"

    """
    The following are based upon actual sequences
    """


    #p1_bsw = 1-(1-p1_data)**alphabet    #The probability of having at least one 1 in binary slepian-wolf
    #p1g1_bsw = 0.85                     #Probability of 1 given 1 in binary LDPC
    #b_swbits_per_obit = 16              #Number of original sequence bits in each bit of the binary slepian-wolf code
    #p1g1_nbs = 0.7            #The probability of "coincidence" in nonbinary code
    #nb_swbits_per_obit = 64   #Effective number of original sequence bits per nonbinary letter.


    #Stats for the actual implementation of the codes
    #b_testparity = 965/1000.0       #Parities I need to send in binary code

    #nb_testparity = 540/1000.0      #Parities I need to send in non-binary code divided by total number of non-parity letters

    print "NON BINARY LOCATION ENTROPY: ", theoretical_nb(coincidence_rate_a, alph)

    print "TOTAL SEQUENCE THEORETICAL ENTROPY:"
    ideal_entropy_alice = theoretical(probability_of_one_in_a,coincidence_rate_a,probability_of_one_in_a,coincidence_rate_a)
    ideal_entropy_bob = theoretical(probability_of_one_in_b,coincidence_rate_b,probability_of_one_in_b,coincidence_rate_b)
    print "Ideal Alice entropy: ",ideal_entropy_alice
    print "\n\nFRAME OCCUPANCY THEORETICAL (APPROX) ENTROPY"
    #Note that the entropy retained is per alphabet bits of the original sequence
    # print "BINARY Letter PROB--->>"
#     print "Alice binary_letter_probabilites ",binary_letter_probabilites
    binary_entropy = theoretical_entropy_transition_matrix(binary_letter_probabilites,b_mat,alph)/alph
    print "Alice Binary_entropy",binary_entropy
#     print "\nPercentage of total entropy recovered:",binary_entropy/ideal_entropy_alice

    print "\n\nFRAME LOCATION THEORETICAL (APPROX) ENTROPY"
#     #Note that the entropy retained is for a large amount of original sequence bits
#     print "nonbinary letter probabilities:", nonbinary_letter_probabilities
    
#     print "transition matrix for locations",transition_matrix_non_binary
    nonbinary_entropy = theoretical_entropy_transition_matrix(nonbinary_letter_probabilities,transition_matrix_non_binary,alph)/nb_bperf

#     print "\nPercentage of total entropy recovered:",nonbinary_entropy/ideal_entropy_bob

    print "\n\nTOTAL PERCENTAGE OF ENTROPY RETAINED:"
# #     print "SHOULD BE LESS THAN ONE AND ARE:"
# #     print "\t %f"%(nonbinary_entropy/ideal_entropy_alice)
#     print "\t %f"%(binary_entropy/ideal_entropy_alice) 
#     print "Theory Alice: %f"%((nonbinary_entropy+binary_entropy)/ideal_entropy_alice)
#     print "Theory Bob:%f"%((nonbinary_entropy+binary_entropy)/ideal_entropy_bob)
    
#     print "\n\n"
    
    return (ideal_entropy_alice,ideal_entropy_bob,binary_entropy,nonbinary_entropy)
def calculate_binary_single_entropy(frame_occupancies, frame_size):
    probability_one = sum(frame_occupancies==1)/float(len(frame_occupancies))
    probability_zero = sum(frame_occupancies==0)/float(len(frame_occupancies))
    
    poisson_factor_one = probability_one*exp(-probability_one)
    poisson_factor_zero = probability_zero*exp(-probability_zero)

    letter_one_probability = poisson_factor_one*probability_one
    letter_zero_probability = poisson_factor_zero*probability_zero
    
    binary_single_entropy = -letter_zero_probability*log2(letter_zero_probability)-letter_one_probability*log2(letter_one_probability)

    return binary_single_entropy

def joint_entropy(alice_frame_occupancies,bob_frame_occupancies):
    probability_one_a = sum(alice_frame_occupancies==1)/float(len(alice_frame_occupancies))
    probability_zero_a = sum(alice_frame_occupancies==0)/float(len(alice_frame_occupancies))
    probability_multi_a = sum(alice_frame_occupancies>1)/float(len(alice_frame_occupancies))
    poisson_factor_one_a = probability_one_a*exp(-probability_one_a)
    poisson_factor_zero_a = probability_zero_a*exp(-probability_zero_a)

    letter_one_probability_a = poisson_factor_one_a*probability_one_a
    letter_zero_probability_a = poisson_factor_zero_a*probability_zero_a
# ------Bob
    probability_one_b = sum(bob_frame_occupancies==1)/float(len(bob_frame_occupancies))
    probability_zero_b = sum(bob_frame_occupancies==0)/float(len(bob_frame_occupancies))   
    poisson_factor_one_b = probability_one_b*exp(-probability_one_b)
    poisson_factor_zero_b = probability_zero_b*exp(-probability_zero_b)

    letter_one_probability_b = poisson_factor_one_b*probability_one_b
    letter_zero_probability_b = poisson_factor_zero_b*probability_zero_b
 
    print "\n\t\t\t Fraction of ones", probability_one_a
    print "\n\t\t\t Fraction of zeros", probability_zero_a
    print "\n\t\t\t Fraction of multi", probability_multi_a
    print "\n\t\t\t TOTAL", probability_zero_a+probability_one_a+probability_multi_a
 
    alice_probs = array([letter_zero_probability_a,letter_one_probability_a])
    bob_probs = array([letter_zero_probability_b,letter_one_probability_b])
    entropy = 0.0

    for alice in alice_probs:
        for bob in bob_probs:
            entropy -=alice*bob*log2(alice*bob)
    return entropy

def shared_entropy(binary_entropy_alice, binary_entropy_bob, joint_entropy):
    return binary_entropy_alice+binary_entropy_bob-joint_entropy

def entropy_calc(binary_string_alice,binary_string_bob, frame_size):
    entropy = 0.0
    frame_occupancies_alice = calculate_frame_occupancy(binary_string_alice, frame_size)
    frame_occupancies_bob = calculate_frame_occupancy(binary_string_bob, frame_size)

    frame_locations_alice = calculate_frame_locations_daniels_mapping(binary_string_alice, frame_occupancies_alice, frame_size)
    frame_locations_bob = calculate_frame_locations_daniels_mapping(binary_string_bob, frame_occupancies_bob, frame_size)

    print "Frame occupancies alice",frame_occupancies_alice
    print "Frame locations alice", frame_locations_alice
    print "Frame occupancies bob",frame_occupancies_bob
    print "Frame locations bob", frame_locations_bob
    
    loc_letter_probabilities_alice = letter_probabilities(frame_locations_alice, frame_size, 1)
    loc_letter_probabilities_bob = letter_probabilities(frame_locations_bob, frame_size, 1)

    occ_letter_probabilities_alice = letter_probabilities(frame_occupancies_alice, frame_size,1)
    occ_letter_probabilities_bob = letter_probabilities(frame_occupancies_bob, frame_size,1)
    
#     print "loc letter probabilities", loc_letter_probabilities_alice
#     print "occ letter probabilities", occ_letter_probabilities_alice

    fl_entropy_alice = calculate_frame_entropy(frame_locations_alice, frame_size)
    fl_entropy_bob = calculate_frame_entropy(frame_locations_bob, frame_size)

    fo_entropy_alice = calculate_frame_entropy(frame_occupancies_alice, frame_size)
    fo_entropy_bob = calculate_frame_entropy(frame_occupancies_bob, frame_size)

    print "Frame locations entropy", fl_entropy_alice
    print "Frame occupancies entropy", fo_entropy_alice


    total_c = intersect1d(binary_string_alice, binary_string_bob)
    
    p1a = float(len(binary_string_alice))/binary_string_alice[-1]
    p1g1a = float(len(total_c))/len(binary_string_alice)
        
    p1b = float(len(binary_string_bob))/binary_string_bob[-1]
    p1g1b = float(len(total_c))/len(binary_string_bob)

    print "\nTheory\n"
    theoretical(p1a, p1g1a, p1b, p1g1b)
    print "\n"

    mutual_frames_with_occupancy_one = logical_and(frame_occupancies_alice==1,frame_occupancies_bob==1)
    maxtag_a = binary_string_alice[-1]

    alice_non_zero_positions_in_frame = frame_locations_alice[mutual_frames_with_occupancy_one]
    bob_non_zero_positions_in_frame   = frame_locations_bob[mutual_frames_with_occupancy_one]
#     mutual_occ_letter_probabilities = letter_probabilities(alice_non_zero_positions_in_frame, frame_size)
#     print alice_non_zero_positions_in_frame
    nb_bperf= maxtag_a/len(alice_non_zero_positions_in_frame)
    nbpl = letter_probabilities(bob_non_zero_positions_in_frame,frame_size, 1)
#     print "non zero pos",alice_non_zero_positions_in_frame
    nbtransmat = transitionMatrix_data2(alice_non_zero_positions_in_frame,bob_non_zero_positions_in_frame,frame_size)
    swtransmat = transitionMatrix_data2(frame_occupancies_alice,frame_occupancies_bob,frame_size)

#     print swtransmat
#     print "-->>", occ_letter_probabilities_bob
    entropy_calculate2(p1a,p1g1a,p1b,p1g1b,frame_size,occ_letter_probabilities_alice,swtransmat,nb_bperf,nbpl,nbtransmat)
    

