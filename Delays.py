
'''
Created on Jun 8, 2016
@author: laurynas
'''
from numpy import concatenate,take,uint64,save, int64,zeros,where,argwhere,intersect1d,load,array,append,savetxt,in1d
import sys
from System import load_data
from numpy import uint8
import ttag
from ttag_delays import getDelay, getDelays
from Statistics import create_binary_string_from_laser_pulses as laser

def remake_coincidence_matrix(coincidence_matrix):
    channels = len(coincidence_matrix[0][:])
    width = channels/2
    height = len(coincidence_matrix[:][0])/2
    matrix = zeros((height,width))
    
    for i in range(height):
        for j in range(width):
            matrix[i][j] = coincidence_matrix[i][channels/2+j]
    return matrix


def check_correlations(aliceTtags,aliceChannels,bobTtags,bobChannels,resolution, A_B_timetags, A_B_channels,channels1,channels2,delays,coincidence_window_radius):

    for delay,ch1,ch2 in zip(delays,channels1,channels2):
        if delay < 0:
            A_B_timetags[A_B_channels == ch2] += (abs(delay)).astype(uint64)
        else:
            A_B_timetags[A_B_channels == ch1] += delay.astype(uint64)
               
    indexes_of_order = A_B_timetags.argsort(kind = "mergesort")
    A_B_channels = take(A_B_channels,indexes_of_order)
    A_B_timetags = take(A_B_timetags,indexes_of_order)      

    buf_num = ttag.getfreebuffer() 
    buffer = ttag.TTBuffer(buf_num,create=True,datapoints = int(5e7))
    buffer.resolution = 260e-12
    buffer.channels = max(A_B_channels)+1
    buffer.addarray(A_B_channels,A_B_timetags)

    
    buf_num = ttag.getfreebuffer()
    bufDelays = ttag.TTBuffer(buf_num,create=True,datapoints = int(5e7))
    bufDelays.resolution = resolution
    bufDelays.channels = max(A_B_channels)+1
    bufDelays.addarray(A_B_channels,A_B_timetags.astype(uint64))
     
    
    
    with_delays = (bufDelays.coincidences((A_B_timetags[-1]-1)*bufDelays.resolution, coincidence_window_radius))
    remade = remake_coincidence_matrix(with_delays)
    
    print "__COINCIDENCES WITH DELAYS ->>\n",remade.astype(uint64)
    
def calculate_delays(aliceTtags,aliceChannels,bobTtags,bobChannels,
                    resolution= 260e-12,
                    coincidence_window_radius = 200-12,
                    delay_max = 1e-6):
    
    channels1 = [0,1,2,3]
    channels2 = [4,5,6,7]
    
    A_B_timetags = concatenate([aliceTtags,bobTtags])
    A_B_channels = concatenate([aliceChannels,bobChannels])

    indexes_of_order = A_B_timetags.argsort(kind = "mergesort")
    A_B_channels = take(A_B_channels,indexes_of_order)
    A_B_timetags = take(A_B_timetags,indexes_of_order)

    buf_num = ttag.getfreebuffer()
    bufN = ttag.TTBuffer(buf_num,create=True,datapoints = int(5e11))
    bufN.resolution = resolution
    bufN.channels = max(A_B_channels)+1
    bufN.addarray(A_B_channels,A_B_timetags)

    coincidences_before = (bufN.coincidences((A_B_timetags[-1]-1)*bufN.resolution, coincidence_window_radius))
    remade = remake_coincidence_matrix(coincidences_before)
    print "__COINCIDENCES BEFORE-->>\n",remade.astype(uint64)

    channel_number  = int(max(append(channels1,channels2)))
    delays = zeros(channel_number)
    k = 0
    for i,j in zip(channels1, channels2):
        delays[i] = (getDelay(bufN,i,j,delaymax=delay_max,time=(A_B_timetags[-1]-1)*bufN.resolution))/bufN.resolution
        print delays[i]
        k+=1
    check_correlations(aliceTtags, aliceChannels, bobTtags, bobChannels, resolution, A_B_timetags, A_B_channels, channels1, channels2, delays, coincidence_window_radius)
    print("Saving delays to file.")
    save("./resultsLaurynas/Delays/delays.npy",delays)
    
if (__name__ == '__main__'):
    alice_channels = [0,1,2,3]
    bob_channels =   [4,5,6,7]
    
    (aliceTtags,aliceChannels) = load_data("alice",alice_channels,100)
    (bobTtags,bobChannels) = load_data("bob",bob_channels,100)
    
    indexes_of_order = aliceTtags.argsort(kind = "mergesort")
    aliceChannels = take(aliceChannels,indexes_of_order)
    aliceTtags = take(aliceTtags,indexes_of_order)
    
    indexes_of_order = bobTtags.argsort(kind = "mergesort")
    bobChannels = take(bobChannels,indexes_of_order)
    bobTtags = take(bobTtags,indexes_of_order)
    
    aliceTtags = aliceTtags[:len(aliceTtags)]
    aliceChannels = aliceChannels[:len(aliceChannels)]
    
    bobTtags = bobTtags[:len(bobTtags)]
    bobChannels = bobChannels[:len(bobChannels)]
    
    calculate_delays(aliceTtags.astype(uint64), aliceChannels.astype(uint8), bobTtags.astype(uint64), bobChannels.astype(uint8),coincidence_window_radius = 200E-12) 

