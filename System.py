'''
Created on Jun 4, 2016

@author: Laurynas Tamulevicius

The system implements QKD protocol for hyper-entanglement experiment,
where the entropy is extracted from polarization and timing events. 

The following system is not fully optimized and alterations can be done to make it work faster.
The weak part of it is error correction as LDPC non-binary codes is somewhat experimental here,
hence belief propagation does not converge for higher alphabet values.

Better parity check matrix could be implemented as well, currently the major amount of errors seems to be
corrected from prior probability matrix, hence Eve does not seem to receive any information as small amount of errors could be corrected
just from prior-probability matrix which is rendered from data exchanged before actual QKD process in order to determine channel statistics.

P.S. The convention for Alice and Bob is changed to Alex and Brad

'''

import threading
from warnings import warn
import time
from os import system
from numpy import *
from PrivacyAmplification import privacy_amplification
from Statistics import *
from ParityCheckMatrixGen import gallager_matrix
from SlepianWolf import encode, check
import timeit
from SW_prep import randomMatrix

'''
  This method returns subframing values used in Brad correction. It's basically, taking laser pulse,
  splitting it into coincidence window size chunks and taking the only mod value of actual timetag in that laser pulse.
  This uses a strong assumption that in laser pulse there's only 1 laser event.
'''
def get_subframe_values(timetags, resolution, sync_period, coincidence_window_radius):
    
    indexes = {}
    timetags = timetags.astype(uint64)
    sync_block_size = int(sync_period / resolution)
    D_block_size = coincidence_window_radius * 2 + 1

    for i in range(len(timetags)):
        ith = (timetags[i] % sync_block_size) % D_block_size
        indexes[int(timetags[i] / sync_block_size)] = ith
    
    return indexes

'''
  The same as above just instead we collect polarization values of each laser pulse event
'''
def get_polarization_corrections(polarizations, timetags, resolution, sync_period, coincidence_window_radius):
    
    indexes = {}
    timetags = timetags.astype(uint64)
    sync_block_size = int(sync_period / resolution)
    
    for i in range(len(timetags)):
        indexes[int(timetags[i] / sync_block_size)] = polarizations[i]
        
    return indexes

'''
  This is used ONLY for testing purposes as it has all timing data where key is laser pulse
  number and value is actual position of event in the laser pulse so adding both you would
  simply get a timetag value.
'''
def get_full_subframe_values(bob_ttags, sync_period, resolution, coincidence_window_radius):
    
    bob_ttag_dict = {}
    sync_block_size = int(sync_period / resolution)
    
    for i in range(len(bob_ttags)):
        ith = (bob_ttags[i] % sync_block_size)
        bob_ttag_dict[int(bob_ttags[i] / sync_block_size)] = ith
        
    return bob_ttag_dict

'''
  Does the actual correction. Takes only laser pulses where both Alex and Brad has a timing events.
  Then Brad uses Alice subrframing values in order to compare it's subframed value, which should be within coincidence
  window (as we care only about coincidences). Hence when the shortest distance between subframe values is found the offset
  is applied to Brad timetag and hence now both Alex and Bob have same timing values.
'''
def do_correction(bob_ttag_dict, bob_pol_dict, bob_dict, alice_dict, coincidence_window_radius, alice_ttag_dict=None):
    
    number_of_bins_in_block = coincidence_window_radius * 2 + 1
    print number_of_bins_in_block
    number_of_bins_in_block_padding = number_of_bins_in_block
    coincidence_ttag_pulses = {}
    coincidence_pol_pulses = {}
    k = 0
    padding_zeros = 0   
    while number_of_bins_in_block_padding != 0:
        number_of_bins_in_block_padding /= 10 
        padding_zeros += 1 
    
    for a_key in alice_dict.keys():

        shift = 0
        if bob_dict.has_key(a_key):
            distance_away = min(abs(number_of_bins_in_block - bob_dict[a_key] + alice_dict[a_key]),
                                abs(alice_dict[a_key] - bob_dict[a_key]),
                                abs(number_of_bins_in_block - alice_dict[a_key] + bob_dict[a_key]))
            
            if distance_away <= coincidence_window_radius:
                if distance_away == abs(alice_dict[a_key] - bob_dict[a_key]):
                    if alice_dict[a_key] - bob_dict[a_key] > 0:
                        shift = distance_away
                    else:
                        shift = -distance_away
                elif distance_away == number_of_bins_in_block - bob_dict[a_key] + alice_dict[a_key]:
                    shift = distance_away
                elif distance_away == number_of_bins_in_block - alice_dict[a_key] + bob_dict[a_key]:
                    shift = -distance_away

                bob_ttag_dict[a_key] += shift
                coincidence_ttag_pulses[a_key] = bob_ttag_dict[a_key]
                coincidence_pol_pulses[a_key] = bob_pol_dict[a_key]
                k += 1
            else:
                bob_ttag_dict[a_key] = -1
                
    return (bob_ttag_dict, coincidence_ttag_pulses, coincidence_pol_pulses)
     
'''
  The method is used to make both Alex and Bob timetag and polarization lists of equal size 
'''
def make_equal_size(alice_thread, bob_thread):

    if (len(alice_thread.ttags) > len(bob_thread.ttags)):
        alice_thread.ttags = alice_thread.ttags[:len(bob_thread.ttags)]
        alice_thread.channels = alice_thread.channels[:len(bob_thread.channels)]
    else:
        bob_thread.ttags = bob_thread.ttags[:len(alice_thread.ttags)]
        bob_thread.channels = bob_thread.channels[:len(alice_thread.channels)]

'''
  Load .npy data and slice it using data_factor variable            
'''
# When data_factor is set to 10 dataset is smaller than all set (i.e. it's about 0.9s of total time)
def load_data(name, channelArray, data_factor):
#     laser_pulse = 260e-12
    sys.stdout.flush()
#     all_ttags = load("./DarpaQKD/"+name+"Ttags.npy")
#     all_ttags = load("./DarpaQKD/"+name+"TtagsREBINNED.npy")
    all_ttags = load("./DarpaQKD/" + name + "TtagsREBINNEDfull.npy")
#     all_ttags = load("./DarpaQKD/"+name+"TtagsBrightAttempt1.npy")
#     all_ttags = load("./DarpaQKD/"+name+"TtagsBrightAttempt2.npy")
#     all_ttags = load("./DarpaQKD/"+name+"TtagsBrightAttempt3.npy")

#     all_channels = load("./DarpaQKD/"+name+"Channels.npy")

#     all_channels = load("./DarpaQKD/"+name+"ChannelsREBINNED.npy")
    all_channels = load("./DarpaQKD/" + name + "ChannelsREBINNEDfull.npy")

#     all_channels = load("./DarpaQKD/"+name+"ChannelsBrightAttempt1.npy")
#     all_channels = load("./DarpaQKD/"+name+"ChannelsBrightAttempt2.npy")
#     all_channels = load("./DarpaQKD/"+name+"ChannelsBrightAttempt3.npy")

    all_ttags = all_ttags[:len(all_ttags) / data_factor]
    all_channels = all_channels[:len(all_channels) / data_factor]
    
    ttags = []
    channels = []
    for ch in channelArray:
        ttags += all_ttags[all_channels == ch].tolist()
        channels += all_channels[all_channels == ch].tolist()
        
    ttags = asarray(ttags, dtype=uint64)
    channels = asarray(channels, dtype=uint8)
                
    indexes_of_order = ttags.argsort(kind="mergesort")
    channels = take(channels, indexes_of_order)
    ttags = take(ttags, indexes_of_order)
#     ttags = (ttags*resolution/laser_pulse).astype(uint64)
    return (ttags, channels)

'''
  Load .csv (text) file from some directory where file has combined information about
  both Alex and Brad timing and polarization events
'''
def load_save_raw_file(dir, alice_channels, bob_channels):
    data = loadtxt(dir)

    channels = data[:, 0]
    timetags = data[:, 1]
    print("Saving Data Arrays")
    sys.stdout.flush()
    
    save("./DarpaQKD/aliceChannelsREBINNED.npy", channels[in1d(channels, alice_channels)])
    save("./DarpaQKD/aliceTtagsREBINNED.npy", timetags[in1d(channels, alice_channels)])
    
    save("./DarpaQKD/bobChannelsREBINNED.npy", channels[in1d(channels, bob_channels)])
    save("./DarpaQKD/bobTtagsREBINNED.npy", timetags[in1d(channels, bob_channels)])
    
'''
  LDPC encoding procedure. It uses parity check matrix (PCM) to encode syndrome values (checksums + redundant data).
  The PCM has dimensions of (parity check nodes (equations) x total bits to encode).
  Column weight and row weight determines the number of connections between edges of the graph
'''    
def LDPC_encode(alice_thread, column_weight=3, row_weight=5):
    total_string_length = len(alice_thread.non_zero_positions)
    
    number_of_parity_check_eqns_gallager = int(total_string_length * column_weight/row_weight)
    
#     alice_thread.parity_matrix = gallager_matrix(number_of_parity_check_eqns_gallager, total_string_length, column_weight, row_weight)
    # bits   - number of encoding strings
    # checks - number of parity check eqns
    # paritis- column weight
    alice_thread.parity_matrix = randomMatrix(total_string_length, number_of_parity_check_eqns_gallager, column_weight)
    alice_thread.syndromes = encode(alice_thread.parity_matrix, alice_thread.non_zero_positions, alice_thread.frame_size)
    
    print "Syndromes", alice_thread.syndromes, len(alice_thread.syndromes), (total_string_length)

'''
    The following method does iterative decoding of original data string chunks. It tries different parity
    check matrices until probabilities converge and bits are corrected
'''    
def LDPC_decode_chunk(alice_thread, bob_thread, number_of_parity_check_eqns, alice_chunk, bob_chunk, attempt, decoder, iterations, frozenFor, column_weight,row_weight = 3):
    attempt = 0
    decoded = False 
    decoded_string = array([])
  
    while not decoded:
        print "==================", attempt, "================================"
        
# =============In case you want to use gallager type matrix================================================================================
#         total_string_length = len(alice_chunk)
#         number_of_parity_check_eqns_gallager = int(total_string_length*column_weight/row_weight)
#         alice_thread.parity_matrix = gallager_matrix(number_of_parity_check_eqns_gallager, total_string_length, column_weight, row_weight)
# ==========================================================================================================================================

        alice_thread.parity_matrix = randomMatrix(len(alice_chunk), number_of_parity_check_eqns, column_weight)
        alice_thread.syndromes = encode(alice_thread.parity_matrix, alice_chunk, alice_thread.frame_size)
        
        #============================SEND PARITY CHECK MATRIX & SYNDROMES =======================================
        bob_thread.syndromes = alice_thread.syndromes
        bob_thread.parity_matrix = alice_thread.parity_matrix
        #========================================================================================================
        
        if attempt == 0:
            transition_matrix = transitionMatrix_data2_python(bob_chunk, alice_chunk , bob_thread.frame_size)
            prior_probability_matrix = sequenceProbMatrix(bob_chunk, transition_matrix)
            
        belief_propagation_system = SW_LDPC(bob_thread.parity_matrix, bob_thread.syndromes, prior_probability_matrix, decoder=decoder, original=alice_chunk)
        decoded_string = belief_propagation_system.decode(iterations=iterations, frozenFor=frozenFor)
        is_not_decoded = sum(check(bob_thread.parity_matrix, decoded_string, bob_thread.frame_size, bob_thread.syndromes) == False)
        if is_not_decoded == 0: 
            decoded = True
        else:
            decoded = False
            
        attempt += 1

    return decoded_string

'''
    Splits string into chunks and iterate them
'''
def LDPC_decode_all_chunks(alice_thread, bob_thread, fraction_parity_checks, decoder='bp-fft', chunk_size=10, iterations=10, frozenFor=5, column_weight=4):
    number_of_chunks = len(alice_thread.non_zero_positions) / chunk_size
    number_of_parity_check_eqns = int(chunk_size * fraction_parity_checks)
    decoded_string_total = array([], dtype=uint8)

    for i in range(number_of_chunks):
        print "\t\t", i, "out of ", number_of_chunks
        start = i * chunk_size
        end = (i + 1) * chunk_size
        alice_chunk = bob_thread.received_string[start:end]
        bob_chunk = bob_thread.non_zero_positions[start:end]

        print "#alice", alice_chunk, "#bob", bob_chunk
        attempt = 0
        decoded_string = LDPC_decode_chunk(alice_thread, bob_thread, number_of_parity_check_eqns, alice_chunk, bob_chunk, attempt, decoder, iterations, frozenFor, column_weight=column_weight)
        print decoded_string == alice_chunk
        decoded_string_total = append(decoded_string_total, decoded_string)
    
    alice_thread.non_zero_positions = alice_thread.non_zero_positions[:len(alice_thread.non_zero_positions) / chunk_size * chunk_size]
    
    return decoded_string_total
'''
  The same as above just used in binary error correction (for polarization bases bits)
'''
def LDPC_binary_encode(alice_thread, column_weight=3, row_weight=4):
    total_string_length = len(alice_thread.bases_string)
    
    number_of_parity_check_eqns_gallager = int(total_string_length * 0.6) 
#     alice_thread.parity_binary_matrix = gallager_matrix(number_of_parity_check_eqns_gallager, total_string_length, column_weight, row_weight)
    alice_thread.parity_binary_matrix = randomMatrix(total_string_length, number_of_parity_check_eqns_gallager, column_weight)
    alice_thread.binary_syndromes = encode(alice_thread.parity_binary_matrix, alice_thread.bases_string, alphabet=2)
    

'''
  Creates transition matrix which is just matrix that combines both Alex and Brad letter probabilities.
  Using non-log decoder might result in divergence.
  Iterations does not converge for not known reason, needs to be checked and fixed.
  Prior probability matrix takes columns from transition matrix that are assigned to appropriate letter in Alex string
  All these vaariables are used in belief propagation system which allows convergence of some most probable value.
'''
def LDPC_decode(bob_thread, alice_thread, decoder='bp-fft', iterations=70, frozenFor=20):
    bob_thread.sent_string = bob_thread.non_zero_positions[:len(bob_thread.received_string)]
    transition_matrix = transitionMatrix_data2_python(bob_thread.sent_string, bob_thread.received_string, bob_thread.frame_size)

    prior_probability_matrix = sequenceProbMatrix(bob_thread.non_zero_positions, transition_matrix)
    print "NON-BINARY TRANS\n\n", transition_matrix
    print "Creating belief propagation system\n"
    belief_propagation_system = SW_LDPC(bob_thread.parity_matrix, bob_thread.syndromes, prior_probability_matrix, decoder=decoder, original=alice_thread.non_zero_positions)
    
    print "Belief propagation system is created\n"
    print "Will be doing decoding using belief prop system\n"
    
    return belief_propagation_system.decode(iterations=iterations, frozenFor=frozenFor)

'''
  Same as above used for binary code correction
'''
def LDPC_binary_decode(bob_thread, alice_thread, decoder='log-bp-fft', iterations=70, frozenFor=10):
    
    bob_thread.sent_binary_string = bob_thread.bases_string[:len(bob_thread.received_binary_string)]
    transition_matrix = transitionMatrix_data2_python(bob_thread.received_binary_string, bob_thread.sent_binary_string, alph=2)
    prior_probability_matrix_binary = sequenceProbMatrix(bob_thread.received_binary_string, transition_matrix)

    print "Creating binary belief propagation system\n"
    belief_propagation_system = SW_LDPC(bob_thread.parity_binary_matrix, bob_thread.binary_syndromes, prior_probability_matrix_binary, original=alice_thread.bases_string, decoder=decoder)
    
    print "Binary belief propagation system is created\n"
    print "Will be doing decoding using binary belief prop system\n"
    
    return belief_propagation_system.decode(iterations=iterations, frozenFor=frozenFor)


'''
  Using existing polarization list creates binary string where 0 represents horizontal/vertical basis
  and 1 - diagonal/anti-diagonal basis
'''
def prepare_bases(channels, channelArray):
    
    bases = zeros(len(channels), dtype=uint8)
    one_diagonal_basis = channelArray[2:]
    
    bases[in1d(channels, one_diagonal_basis)] = 1
    
    return bases    

'''
  Main thread class for Alex and Bob which are instances of it.
  The threadig is used in order to exploit parallelism as data processing takes a lot of time
  and we have to do that for both Alex and Brad.
'''
class PartyThread(threading.Thread):
    
    def __init__(self, resolution, name, channelArray, coincidence_window_radius, delay_max, data_factor):
        threading.Thread.__init__(self)
        self.running = True
        self.name = name
        self.data_factor = data_factor
        self.resolution = resolution
        self.channelArray = channelArray
        self.coincidence_window_radius = coincidence_window_radius
        self.event = threading.Event()
        self.frame_size = 2
        self.race_flag = False
        self.delay_max = delay_max
        self.event.set()
        self.full_dict = array([])
        self.corrected_dict = array([])
        self.corrected_pol_dict = array([])
    
    '''
      As event is cleared no other thread will be blocked. Simply used as a flag.
    '''    
    def do_clear(self):
        self.event.clear()
        
    '''
    Setting the event
    '''
    def do_set(self):
        self.event.set()
        
    '''
      Main method of the thread where actual processing happens
    '''
    def run(self):
        
        while self.running:
            
#       This should be uncommented if raw text file has to be processed first  

#             print self.name.upper()+" : Reading.csv files and converting to .npy\n"
#             system("python ./DataProcessing.py "+self.raw_file_name+" "+self.name)

            print self.name.upper() + ": Loading .npy data\n"
            (self.ttags, self.channels) = load_data(self.name, self.channelArray, data_factor)
            print "#of ttags", len(self.ttags), "where last is ", self.ttags[-1]
            print "TOTAL TIME: ", self.ttags[-1] * 260e-12, " in seconds"
            
            print self.name.upper() + ": Loading delays\n"
            self.delays = load("./resultsLaurynas/Delays/delays.npy")
            print self.delays


            print self.name.upper() + ": Applying Delays"

            self.ttags = self.ttags.astype(int64)

            for delay, ch1 in zip(self.delays, self.channelArray):
                if delay < 0 and self.name == "bob":
                    self.ttags[self.channels == ch1] += (abs(delay)).astype(uint64)
                elif delay >= 0 and self.name == "alice":
                    self.ttags[self.channels == ch1] += delay.astype(uint64)


            # Sorting all data with delays
            indexes_of_order = self.ttags.argsort(kind="quicksort")
            self.channels = take(self.channels, indexes_of_order)
            self.ttags = take(self.ttags, indexes_of_order)
            
            self.ttags = self.ttags.astype(uint64)
            self.channels = self.channels.astype(uint8)

            
            print self.name.upper() + ": FINISHED with data. Will notify main\n"        


            print self.name.upper() + ": Waiting for OPTIMAL FR SIZE\n" 
            
# ==========TYPICAL BLOCK TO WAIT FOR MAIN and after RESET SELF AGAIN
            self.do_clear()
            while main_event.is_set():
                pass
            self.race_flag = True

# ===================================================================   

            
            print self.name.upper() + ": Calculating frame occupancies and locations...\n"
            self.frame_occupancies = calculate_frame_occupancy(self.ttags, self.frame_size)

#             self.frame_locations = calculate_frame_locations_daniels_mapping(self.ttags, self.frame_occupancies, self.frame_size)
            (self.frame_locations, self.frame_location_channels) = calculate_frame_locations_for_single_occ(self.ttags, self.channels, self.frame_occupancies, self.frame_size)
            
            print self.name.upper() + ": FINISHED Calculating frame occupancies and locations. WILL RELEASE MAIN\n"
            self.do_clear()

            self.running = False
               
if __name__ == '__main__':

    
    start = time.time()
# ============ PARAMETERS ===========================================

    raw_file_dir = "./DarpaQKD/Alice1_Bob1.csv"
    alice_channels = [0, 1, 2, 3]
    bob_channels = [4, 5, 6, 7]
    
#   Uncomment if raw text file has to be processed first
#     load_save_raw_file(raw_file_dir, alice_channels, bob_channels)
    
    
    set_printoptions(edgeitems=20)
    resolution = 260e-12
    
#   perfect window for bright data is 3905e-12  
    coincidence_window_radius = 200 - 12
    
    delay_max = 1e-5
#     sync_period = 63 * 260e-12
    announce_fraction = 1.0
    announce_binary_fraction = 1.0
#     D_block_size = int(coincidence_window_radius / resolution) * 2 + 1
    data_factor = 1000
    optimal_frame_size = 16
    column_weight = 5
    row_weight = 32
    fraction_parity_checks = 0.6
    
#     padding_zeros = 0
#     while D_block_size != 0:
#         D_block_size /= 10 
#         padding_zeros += 1 
        
#  ===================================================================   
    
    alice_event = threading.Event()
    alice_event.set()
   
    bob_event = threading.Event()
    bob_event.set()
    
     
    alice_thread = PartyThread(resolution,
                               name="alice",
                               channelArray=alice_channels,
                               coincidence_window_radius=coincidence_window_radius,
                               delay_max=delay_max,
                               data_factor=data_factor)
    
    bob_thread = PartyThread(resolution,
                             name="bob",
                             channelArray=bob_channels,
                             coincidence_window_radius=coincidence_window_radius,
                             delay_max=delay_max,
                             data_factor=data_factor)
       
    main_event = threading.Event()
    main_event.set()
    alice_thread.start()
    bob_thread.start()

    print "MAIN: will wait till AB finished loading data."
    while alice_thread.event.is_set() or bob_thread.event.is_set():
        pass

    print "MAIN: STATISTICS: "
    (alice, bob, alice_chan, bob_chan) = (alice_thread.ttags, bob_thread.ttags, alice_thread.channels, bob_thread.channels)
# ===================== OPTIMAL FRAME SIZE EXTRACTION ===================================================================
#     
#     statistics = calculateStatistics(alice,bob,alice_chan,bob_chan, resolution)
#     print statistics
#     max_shared_binary_entropy = max(statistics.values())
#     optimal_frame_size = int(list(statistics.keys())[list(statistics.values()).index(max_shared_binary_entropy)])
#     print "MAIN: The maximum entropy was found to be ",max_shared_binary_entropy," with frame size: ",optimal_frame_size
# =======================================================================================================================
    alice_thread.frame_size = optimal_frame_size
    bob_thread.frame_size = optimal_frame_size

    print "MAIN: Optimal size calculated and set for both threads, release THEM!"
    main_event.clear()
    alice_thread.do_set()
    bob_thread.do_set()

#   Race flag is used as there's data race among threads so we have to sync them  
    while not(alice_thread.race_flag and bob_thread.race_flag):
        pass
    main_event.set()
    
    

    while alice_thread.event.is_set() or bob_thread.event.is_set():
        pass
    
    total = 0
#     for a_ch in alice_thread.channelArray:
#         for b_ch in bob_thread.channelArray: 
#             numb = len(intersect1d(bob_thread.ttags[bob_thread.channels == b_ch], alice_thread.ttags[alice_thread.channels == a_ch]))
# #             print "Coincidences before correction between",a_ch,"-",b_ch,numb
#             total+=numb
#     print "TOTAL COINCIDENCES BEFORE",total,"%",total/float(len(alice_thread.ttags))
#     
# #   Gathers data for correction  
#     total_ttags = 0
#     for bob_full_dict, bob_correction, alice_correction, alice_full_dict, bob_pol_correction in zip(bob_thread.full_dict,
#                                                                                                     bob_thread.correction_array,
#                                                                                                     alice_thread.correction_array,
#                                                                                                     alice_thread.full_dict,
#                                                                                                     bob_thread.pol_correction_array):
# 
#         correction, coincidence_ttag_pulses,coincidence_pol_pulses = do_correction(bob_full_dict,
#                                                                                    bob_pol_correction,
#                                                                                    bob_correction,
#                                                                                    alice_correction, 
#                                                                                    int(bob_thread.coincidence_window_radius/bob_thread.resolution), 
#                                                                                    alice_ttag_dict = alice_full_dict)
# 
#         bob_thread.corrected_dict = append(bob_thread.corrected_dict, coincidence_ttag_pulses)
#         bob_thread.corrected_pol_dict = append(bob_thread.corrected_pol_dict, coincidence_pol_pulses)
#         
#         total_ttags +=len(coincidence_ttag_pulses.keys())
# 
# # ======================= FOR DEBUGGING ====================================================       
#  
#     A_B_channels = concatenate([alice_thread.channels,bob_thread.channels])
#     A_B_timetags = concatenate([alice_thread.ttags,bob_thread.ttags])
#   
#     indexes_of_order = A_B_timetags.argsort(kind = "mergesort")
#     A_B_channels = take(A_B_channels,indexes_of_order)
#     A_B_timetags = take(A_B_timetags,indexes_of_order)
#    
#     A_B_channels.reshape(len(A_B_channels),1)
#     A_B_timetags.reshape(len(A_B_timetags),1)
#     savetxt("./DarpaQKD/Alice1_Bob1_with_delaysREBINNED.txt",np.c_[A_B_channels,A_B_timetags], fmt='%2s %10d')
#     load_save_raw_file("./DarpaQKD/Alice1_Bob1_with_delaysREBINNED.txt", alice_channels, bob_channels)
# 
# # ========================================================================================
#     
#     corrected_ttags = zeros(total_ttags, dtype = uint64)
#     corrected_pol = zeros(total_ttags, dtype = uint64)
#     sync_block_size = int(bob_thread.sync_period/bob_thread.resolution)
#     i=0
#     
# #   Applies corrections (When I was testing it, it was able to recover 97% of coincidences)
#     for corrected_dict,corrected_dict_pol in zip(bob_thread.corrected_dict, bob_thread.corrected_pol_dict):
#         for key in corrected_dict.keys():
#             corrected_ttags[i] = int(str(key))*(sync_block_size)+int(float(str(corrected_dict[key])))
#             corrected_pol[i] = corrected_dict_pol[key]
#             i+=1
# 
# #   Sorts the data         
#     indexes_of_order = corrected_ttags.argsort(kind = "quicksort")
#     corrected_pol = take(corrected_pol,indexes_of_order)
#     corrected_ttags = take(corrected_ttags,indexes_of_order)    
#     
#     print "MAIN: FRACTION OF CORRECTLY CORRECTED COINCIDENCES:",len(intersect1d(corrected_ttags, alice_thread.ttags))/float(len(corrected_ttags))
#     
#     bob_thread.ttags = corrected_ttags
#     bob_thread.channels = corrected_pol
#     
    
    total = 0
    for a_ch in alice_thread.channelArray:
        for b_ch in bob_thread.channelArray: 
            numb = len(intersect1d(bob_thread.ttags[bob_thread.channels == b_ch], alice_thread.ttags[alice_thread.channels == a_ch]))
#             print (intersect1d(bob_thread.ttags[bob_thread.channels == b_ch], alice_thread.ttags[alice_thread.channels == a_ch]))
            total += numb
    print "TOTAL COINCIDENCES AFTER", total, "%", total / float(len(alice_thread.ttags))
    
    

    print "MAIN: BOTH finished calculating frame occ and loc will do mutual frames\n"

# ===================MAKES DATASETS EQUAL SIZE==================================================

    (alice_thread.frame_occupancies, bob_thread.frame_occupancies) = make_data_string_same_size(alice_thread.frame_occupancies, bob_thread.frame_occupancies)
    (alice_thread.frame_locations, bob_thread.frame_locations) = make_data_string_same_size(alice_thread.frame_locations, bob_thread.frame_locations)
    (alice_thread.frame_location_channels, bob_thread.frame_location_channels) = make_data_string_same_size(alice_thread.frame_location_channels, bob_thread.frame_location_channels)
# ==============================================================================================

    mutual_frames_with_occupancy_one = logical_and(alice_thread.frame_occupancies == 1, bob_thread.frame_occupancies == 1)
    mutual_frames_with_multiple_occ = logical_and(alice_thread.frame_occupancies > 1, bob_thread.frame_occupancies > 1)

#   Takes only frames with occupancy of one
    print "MUTUAL FRAMES WITH OCC 1\n", alice_thread.frame_locations[mutual_frames_with_occupancy_one]
    print bob_thread.frame_locations[mutual_frames_with_occupancy_one]
    print "MAIN: FRACTION OF FRAMES WITH MULTIPLE OCCUPANCY: ", sum(mutual_frames_with_multiple_occ) / float(len(alice_thread.frame_occupancies))
    alice_non_zero_positions_in_frame = alice_thread.frame_locations[mutual_frames_with_occupancy_one]
    alice_non_zero_positions_in_frame_channels = alice_thread.frame_location_channels[mutual_frames_with_occupancy_one]
    
    bob_non_zero_positions_in_frame = bob_thread.frame_locations[mutual_frames_with_occupancy_one]
    bob_non_zero_positions_in_frame_channels = bob_thread.frame_location_channels[mutual_frames_with_occupancy_one]
    
    print "MAIN: Total number of frames ", len(alice_thread.frame_occupancies), " where mutual non zero frames ", len(alice_non_zero_positions_in_frame)
    
#   Calculates binary string with bases values and find mutual bases with same values   
    alice_thread.bases_string = prepare_bases(alice_non_zero_positions_in_frame_channels, alice_thread.channelArray)
    bob_thread.bases_string = prepare_bases(bob_non_zero_positions_in_frame_channels, bob_thread.channelArray)
    print "Bases strings"
    print alice_non_zero_positions_in_frame_channels
    print bob_non_zero_positions_in_frame_channels
    print alice_thread.bases_string
    print bob_thread.bases_string
    
    mutual_bases = where(alice_thread.bases_string == bob_thread.bases_string)
    
#   Estimates QBER where it is determined from non-mathcing bases  TODO: should be determined only from announced fraction
    QBER = 1 - len(mutual_bases[0]) / float(len(bob_thread.bases_string))
    print "ERROR IN BASES (QBER)", QBER
    
    if QBER * 100 > 21:
        warn("The QBER is greater than 21%!!!!!")


    print "MAIN: Now I will throw out all different-polarization coincidences\n"

    alice_thread.non_zero_positions = alice_non_zero_positions_in_frame[mutual_bases]
    alice_thread.non_zero_positions_channels = alice_non_zero_positions_in_frame_channels[mutual_bases]
        
    bob_thread.non_zero_positions = bob_non_zero_positions_in_frame[mutual_bases]
    bob_thread.non_zero_positions_channels = bob_non_zero_positions_in_frame_channels[mutual_bases]
    
#   The LDPC matrices start from 0 so alphabet size must be reduced by one
    bob_thread.non_zero_positions -= 1
    alice_thread.non_zero_positions -= 1

    print "MAIN: MUTUAL FRAME LOCATIONS: ", sum(bob_thread.non_zero_positions == alice_thread.non_zero_positions), " out of ", len(alice_thread.non_zero_positions), " % ", float(sum(bob_thread.non_zero_positions == alice_thread.non_zero_positions)) / len(alice_thread.non_zero_positions)
    print "MAIN: MUTUAL FRAME LOCATION CHANNELS", sum(bob_thread.non_zero_positions_channels == alice_thread.non_zero_positions_channels + 4), "out of ", len(alice_thread.non_zero_positions_channels)
    print "LENGTH of Alice non-secret timing key", len(alice_thread.non_zero_positions)
    print "LENGTH of Bob non-secret timing key", len(bob_thread.non_zero_positions)
#  =======================Will be announcing some part of the string==================================
    print "MAIN: Alice and Bob are now ANNOUNCING " + str(announce_fraction) + " of their frame position strings\n"
    alice_thread.received_string = bob_thread.non_zero_positions[:int(len(bob_thread.non_zero_positions) * announce_fraction)]
    bob_thread.received_string = alice_thread.non_zero_positions[:int(len(alice_thread.non_zero_positions) * announce_fraction)]

    alice_thread.received_binary_string = bob_thread.bases_string[:int(len(bob_thread.bases_string) * announce_binary_fraction)]
    bob_thread.received_binary_string = alice_thread.bases_string[:int(len(alice_thread.bases_string) * announce_binary_fraction)]
    print len(alice_thread.non_zero_positions), len(bob_thread.non_zero_positions)
    print "MAIN: The error in the timing key before correcting", sum(bob_thread.non_zero_positions != alice_thread.non_zero_positions) / len(alice_thread.non_zero_positions)
    print "MAIN: SUCCESFULLY ANNOUNCED WILL RELEASE THREADS\n"
    main_event.clear()
    
    print "MAIN: LDPC: Encoding both NON-BINARY AND BINARY "
    
    LDPC_binary_encode(alice_thread)

#=============Sending syndrome values and parity check matrix=====

#     print "MAIN: Sending syndrome values and parity check matrix\n"
#     
#     bob_thread.syndromes = alice_thread.syndromes
#     bob_thread.parity_matrix = alice_thread.parity_matrix
#     
    bob_thread.binary_syndromes = alice_thread.binary_syndromes
    bob_thread.parity_binary_matrix = alice_thread.parity_binary_matrix
    
#==================================================================

    print "MAIN: Will be trying to decode and correct the string\n"
    print "MAIN: BINARY DECODING\n"
    bob_thread.bases_string = LDPC_binary_decode(bob_thread, alice_thread)
    print "MAIN: NON-BINARY DECODING\n"

    bob_thread.non_zero_positions = LDPC_decode_all_chunks(alice_thread, bob_thread, fraction_parity_checks=0.6, chunk_size=40, iterations=10, column_weight=3)

    print "MAIN: Key length", len(alice_thread.non_zero_positions), "and number of bits", (optimal_frame_size - 1).bit_length()
    print "MAIN: NON-SECRET-KEY-RATE: MBit/s", ((((optimal_frame_size - 1).bit_length() * len(alice_thread.non_zero_positions)) + (len(alice_thread.bases_string))) / (alice_thread.ttags[-1] * 260e-12)) / 1e6
    print "Alice errors at",alice_thread.non_zero_positions[where(alice_thread.non_zero_positions != bob_thread.non_zero_positions)]
    print "Bob errors at",bob_thread.non_zero_positions[where(alice_thread.non_zero_positions != bob_thread.non_zero_positions)]
    if (sum(alice_thread.non_zero_positions == bob_thread.non_zero_positions)) / float(len(alice_thread.non_zero_positions)) == 1.0 :
        print "CONGRATULATIONS! ALEX AND BRAD TIMING STRINGS ARE MATCHING\n"
        
    alice_key = append(alice_thread.bases_string, alice_thread.non_zero_positions)
    bob_key = append(bob_thread.bases_string, bob_thread.non_zero_positions)
    
    print "MAIN: Secret key matches with fraction of:", (sum(alice_key == bob_key)) / float(len(alice_key))
    
    eves_bits = int(QBER * len(alice_thread.bases_string)) + int(len(alice_thread.non_zero_positions) * fraction_parity_checks)
    
    print "MAIN: NON SECRET BITS", len(alice_key), "EVE KNOWS", eves_bits, "BITS"
    print "MAIN: PRIVACY AMPLIFICATION\n"
    (alice_thread.seed, alice_key) = privacy_amplification(alice_key, len(alice_key) - eves_bits, alice_thread.frame_size)
    
#=============Exchanging seed for random hashing function=====

    bob_thread.seed = alice_thread.seed
    
#============================================================

    bob_key = privacy_amplification(bob_key, len(bob_key) - eves_bits, bob_thread.frame_size, bob_thread.seed)

    print "MAIN: Secret key matches after PA: ", sum(alice_key == bob_key) / float(len(alice_key))
    print "MAIN: SECRET BITS:", len(alice_key)
    print "MAIN: SECRET-KEY-RATE: MBit/s", ((((optimal_frame_size - 1).bit_length() * (len(alice_thread.non_zero_positions) - int(len(alice_thread.non_zero_positions) * fraction_parity_checks))) + int(len(alice_thread.bases_string) * (1 - QBER))) / (alice_thread.ttags[-1] * 260e-12)) / 1e6
 
    savetxt("./Secret_keys/alice_secret_key1.txt", alice_key, fmt="%2d")
    savetxt("./Secret_keys/bob_secret_key1.txt", bob_key, fmt="%2d")
    
    end = time.time()
    print "MAIN: DUARTION", (end - start)
    
