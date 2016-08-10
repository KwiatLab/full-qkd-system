from pylab import *
#Fit to the given curve
from numpy import *
from scipy.optimize import curve_fit
import ttag

from itertools import product

rc('text', usetex=True)
rc('font', family='serif')

# Define model function to be used to fit to the data above:

def gauss(x, *p):
    A, mu, sigma = p
    return A*exp(-(x-mu)**2/(2.*sigma**2))


#Assume that the delay is < 1ms
#Also, just assume that max one is the correct delay - a gaussian fit might be better
#   for certain things
def getDelay(buf,channel1,channel2,initialdelay1=0.0,initialdelay2=0.0,delaymax = 0.0000001,time=1.0):

    bins = int(delaymax/buf.resolution)*2
#     print delaymax,"coincidence window radius "," window length in bins ",bins, time, " time back "
    corr = buf.correlate(time,delaymax,bins,channel1,channel2,channel1delay=initialdelay1,channel2delay=initialdelay2)
#     print len(corr),corr
    #Now, we have a way to fit to gaussian - set initial parameters
    mu = argmax(corr)
    sigma = 5
    A = max(corr)
#     
#     try:
#         popt,pcov = curve_fit(gauss,range(bins),corr,p0=(A,mu,sigma))
#         print(channel1,channel2,"FIT: (A,mu,sigma)=",popt)
#         return (popt[1]-len(corr)/2)*buf.resolution
#     except:
    return (mu-len(corr)/2)*buf.resolution


#This function cannot be used on huge buffers, since it creates a copy of the entire dataset
def getPossibleInitialDelays(buf,syncChannel1,syncChannel2):
    channels,singles = buf[:]

    possibilities1= where(channels==syncChannel1)[0]
    possibilities2= where(channels==syncChannel2)[0]

    delays=[]

    for i1,i2 in product(possibilities1,possibilities2):
        delays.append(singles[i2]-singles[i1])

    return delays

def getDelays(buf,channels1,channels2,initialdelay2=0.0,delays1=None,delays2=None,delaymax=0.0000001,time=1.0):

    if (delays1==None):
        delays1=zeros(len(channels1))
    if (delays2==None):
        delays2=ones(len(channels2))*initialdelay2

    #First set all of channels2 delays
    for i in range(len(delays2)):
        delays2[i] += getDelay(buf,channels1[0],channels2[i],delays1[0],delays2[i],delaymax=delaymax,time=time)

    #Next, set all of delays for channels1
    for i in range(1,len(delays1)):
        delays1[i] -= getDelay(buf,channels1[i],channels2[0],delays1[i],delays2[0])

    return (delays1,delays2)
"""
def plotAll(buf,channels1,channels2,delays1,delays2):
    bins = 100
    length = bins/2*buf.resolution
    dist=(array(range(bins))-bins/2)*buf.resolution
    f,ax=subplots(len(channels1),sharex=True)
    plots = []
    for i in range(len(channels1)):
        for j in range(len(channels2)):
            ax[i].set_ylabel("Channel "+str(i+1))
            cor=buf.correlate(1.0,length,bins,channels1[i],channels2[j],channel1delay=delays1[i],channel2delay=delays2[j])
            #Now, we have a way to fit to gaussian - set initial parameters
            mu = argmax(cor)
            sigma = 5
            A = max(cor)
            popt,pcov = curve_fit(gauss,range(bins),cor,p0=(A,mu,sigma))
            plots.append(ax[i].plot(dist,cor,linewidth=2,label=r"Channel "+str(j+1) + r" (A="+str(int(round(popt[0])))+r" $\sigma$="+str(round(popt[2]*buf.resolution*1e12,2))+"ps)"))
    ax[0].set_title("Correlations Between Alice and Bob's Channels")
    f.subplots_adjust(hspace=0)
    ax[0].legend()
"""
# buf = ttag.TTBuffer(0)
# d = getPossibleInitialDelays(buf,0,6)
# 
# channels1=[2,3,4,5]
# channels2=[8,9,10,11]
# 
# d1,d2 = getDelays(buf,channels1,channels2,d[0])
# 
# print("Second Round of Delay finding")
# d1,d2 = getDelays(buf,channels1,channels2,delays1=d1,delays2=d2,delaymax=buf.resolution*100)
# 
# print("Preparing Correlation Plot")
# graphs.plotABCorrelations(buf,channels1,channels2,d1,d2)
# user=input("Looks good? (y/n):")
# if (user=="y"):
#     print("Creating Syncd Data...")
#     channels,timetags = buf[:]
# 
#     print("- Applying Delays")
#     for i in range(len(channels1)):
#         timetags[channels==channels1[i]]-=d1[i]
#     for i in range(len(channels2)):
#         timetags[channels==channels2[i]]-=d2[i]
# 
#     print("- Extracting Alice and Bob")
#     allWanted = (channels==channels1[0])
#     for i in range(1,len(channels1)):
#         allWanted= logical_or(allWanted,channels==channels1[i])
#     for c in channels2:
#         allWanted = logical_or(allWanted,channels==c)
# 
#     channels = channels[allWanted]
#     timetags = timetags[allWanted]
# 
#     c1b = []
#     c2b = []
#     for c in range(len(channels1)):
#         c1b.append(channels==channels1[c])
#     for c in range(len(channels2)):
#         c2b.append(channels==channels2[c])
# 
#     for i in range(len(channels1)):
#         channels[c1b[i]]=i
#     for i in range(len(channels2)):
#         channels[c2b[i]]=i+len(channels1)
# 
#     """
#     #
#     ##WTF: This code causes a segfault later! I don't even... I don't have time right now to fix it.
#     #
#     print("- Finding intersect of data from both time taggers")
#     #Find the first and last time tags of the two time taggers
#     #   and then take only the intersecting sets
#     c1I = c1b[0]
#     for i in range(1,len(c1b)):
#         c1I= logical_or(c1I,c1b[i])
#     c2I = c2b[0]
#     for i in range(1,len(c2b)):
#         c2I= logical_or(c2I,c2b[i])
#     
#     ttmin = logical_and(timetags > min(timetags[c1I]),timetags > min(timetags[c2I]))
#     ttmax = logical_and(timetags < max(timetags[c1I]),timetags < max(timetags[c2I]))
#     tttot = logical_and(ttmin,ttmax)
#     timetags = timetags[tttot]
#     channels = channels[tttot]
#     """
# 
#     print("- Sorting")
#     #Sort again to make sure everything is fine
#     order = timetags.argsort()
#     timetags = take(timetags,order)
#     channels = take(channels,order)
# 
#     print(len(channels),len(timetags))
# 
# 
#     print("- Creating Buffer")
#     buf_num = ttag.getfreebuffer()
# 
#     print("- Opening Buffer",buf_num)
#     buf2 = ttag.TTBuffer(buf_num,create=True,datapoints = len(channels))
# 
#     print("- Setting Properties")
#     buf2.resolution = buf.resolution
#     buf2.channels = max(channels)+1
#     print("- > Resolution:",buf2.resolution)
#     print("- > Channels:",buf2.channels)
# 
#     print("- Converting timetags to BIN format")
#     #First: Make the smallest tag 0 to avoid possible negatives
#     timetags-=timetags[0]
#     #Convert to bins
#     timetags = (around((timetags)/buf2.resolution)).astype(uint64)
# 
#     print(timetags,channels)
#     print("- Adding to Buffer")
#     buf2.addarray(channels,timetags)
# 
#     print("\nBuffer",buf_num,"Ready.\n\nWhen done, press ENTER to clean up.")
# 
#     input()