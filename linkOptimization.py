#With this program you can compare classical- with quantum communication techniques when transmitting over links containoing linear amplifiers.
#In particular you can calculate the energy that you can save by reducing the gain of amplifiers along the link when using quantum communication techniques instead of classical ones.
#The following license applies:
##MIT License
##Copyright (c) [2022] [Janis Nötzel and Matteo Rosati]
##Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
##The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
##THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from datetime import datetime
import time
import math
from multiprocessing import Process, Queue, Pool
import multiprocessing
import warnings


def fxn():
    warnings.warn("overflow encountered in long_scalars", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

ACCURACY = 12


# first we define the channel models 

def g (x):
    """
    This function calculates the Gordon function
    :param x: a non-negative number
    """
    if x>0:
        return np.log2( 1 + x) + x*np.log2(1 + 1/x)
    else:
        return np.log2( 1 + x)

def dg (x):
    """
    This function calculates the derivative of the Gordon function
    :param x: a non-negative number
    """
    if x>0:
        return np.log2( 1 + 1/x)
    else:
        return np.power(10,10) # well, that's not really infinity but should suffice

# this is the spectral efficiency according to the Holevo formula
def sHol (tau, ns, noise):
    """
    This function calculates the Holevo capacity (Optimal Joint Detection Receiver, OJDR) of the lossy and noisy channel https://doi.org/10.1038/nphoton.2014.216
    :param tau  : a number between zero and one that models the transmittivity (tau=0 means all photons are lost during transmission)
    :param ns   : a non-negative number specifying the (expected) number of signal photons per pulse at the transmitter
    :param noise: a non-negative number specifying the (expected) number of noise photons per pulse
    """
    return g( ns*tau + noise ) - g( noise)

# this is the spectral efficiency according to the Shannon Hartley formula    
def sSha (tau, ns, noise):
    """
    This function calculates the Shannon capacity (Optimal Single Shot Receiver, OSSR) of the lossy and noisy channel https://doi.org/10.1038/nphoton.2014.216
    :param tau  : a number between zero and one that models the transmittivity (tau=0 means all photons are lost during transmission)
    :param ns   : a non-negative number specifying the (expected) number of signal photons per pulse at the transmitter
    :param noise: a non-negative number specifying the (expected) number of noise photons per pulse
    """
    return np.log2( 1 + tau*ns/( 1 + noise ) )

def sHomodyne (tau, ns, noise):
    """
    This function calculates the Shannon capacity using homodyne detectionn of the lossy and noisy channel https://doi.org/10.1038/nphoton.2014.216
    :param tau  : a number between zero and one that models the transmittivity (tau=0 means all photons are lost during transmission)
    :param ns   : a non-negative number specifying the (expected) number of signal photons per pulse at the transmitter
    :param noise: a non-negative number specifying the (expected) number of noise photons per pulse
    """
    return np.log2( 1 + 4*tau*ns/( 1 + 2*noise ) )/2

def vGainNoise ( attenuation, gainList, length ) :
    """
    Calculate the noise level after the last amplifier. The receiver receives an attenuatiated version of this noise!
    :param attenuation : a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :param gainList    : a list of gains that are applied to the amplifiers which are assumed as evenly spaced
    :param length      : positive real modelling total length of the fiber in [km]
    """
    noiseval = 0
    segments = len(gainList) + 1
    # now calculate the transmittivity per segment, d
    d = np.exp( -attenuation*length/segments )
    for i in range(len(gainList)):
        if gainList[i] < 1: # make sure all gains stay above 1, otherwise throw exception
            raise Exception("gain number ",i," is smaller than 1")
        noiseval = gainList[i]*d*noiseval + gainList[i] - 1
    return noiseval

def expPump (attenuation, length, gainList, ns) :
    """
    Computes the energy cost associated to a link with given parameters, up to a constant factor
    This version considers the additional loss that one faces when sending the energy to the EDFA.
    The formula is taken from the accompanying paper
    :param attenuation: a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :param gainList   : a list of gains that are applied to the amplifiers which are assumed as evenly spaced
    :param length     : positive real modelling total length of the fiber in [km]
    :ns               : non-negative real modelling the expected signal energy per pulse at the transmitter in photons
    """
    segments = len(gainList) + 1
    d = np.exp( -attenuation*length/segments )    
    return ns*sum([  np.power( 1/d, np.min([i+1, segments - i]))*( gainList[i] - 1 )*( np.power(d,i+1)*np.prod([gainList[j] for j in range(i)]) + 1) for i in range(segments -1) ])

def ePump (attenuation, length, gainList, ns) :
    """
    Computes the energy cost associated to a link with given parameters, up to a constant factor
    The formula is taken from the accompanying paper
    :param attenuation: a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :param gainList   : a list of real numbers modelling amplifier gains [g[0], .... ] each larger than or equal to 1 of the evenly spaced amplifiers
    :param length     : positive real modelling total length of the fiber in [km]
    :ns               : non-negative real modelling the expected signal energy per pulse at the transmitter in photons
    """
    segments = len(gainList) + 1
    d = np.exp( -attenuation*length/segments )    
    return ns*sum([  ( gainList[i] - 1 )*( np.power(d,i+1)*np.prod([gainList[j] for j in range(i)]) + 1) for i in range(segments -1) ])

def rePump (attenuation, length, gainList, ns) :
    """
    Computes the energy cost associated to a link with given parameters, up to a constant factor
    The formula is the relaxation which overestimates the energy consumption and is taken from the accompanying paper
    :param attenuation: a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :param gainList   : list of real numbers modelling amplifier gains [g[0], .... ] each larger than or equal to 1 of the evenly spaced amplifiers
    :param length     : positive real modelling total length of the fiber in [km]
    :ns               : non-negative real modelling the expected signal energy per pulse at the transmitter in photons
    """
    segments = len(gainList) + 1
    d = np.exp( -attenuation*length/segments )    
    return sum( [  ( gainList[i] - 1 )*( d*ns + 1 ) for i in range(len(gainList)) ] )


def gainMatrix( gain, tau ):
    """
    :param gain: a real number satisfying gain > 1
    :tau       : transmittivity is a number between 0 and 1
    """
    return np.array([[ tau*gain, gain -1 ], [0, 1]], dtype='int64')

def dGainMatrix( tau ):
    """
    derivative of the function gainMatrix with respect to the gain
    :param gain: a real number satisfying gain > 1
    :tau       : transmittivity is a number between 0 and 1
    """
    return np.array([[ tau, 1 ], [0, 0]],dtype='int64')

def eGradient (attenuation, gainList, length, ns ) :
    """
    derivative of the gradient of the total energy with respect to the gains
    :param attenuation: a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :param gainList   : list of real numbers modelling amplifier gains [g[0], .... ] each larger than or equal to 1 of the evenly spaced amplifiers
    :param length     : positive real modelling total length of the fiber in [km]
    :ns               : non-negative real modelling the expected signal energy per pulse at the transmitter in photons
    """
    segments = len(gainList) + 1
    d = np.exp( - attenuation*length/segments )
    factor = [ ns*np.prod( [gainList[j] for j in range(i)] ) for i in range(segments-1)]
    term2 = [ sum([np.power(d,j+1)*(gainList[j] - 1) for j in range(i+1,segments-1)])/gainList[i]  for i in range(segments-1)]
    return [ factor[i]*(np.power(d,i+1) + term2[i]) for i in range(len(gainList))] 

def expGradient (attenuation, gainList, length, ns ) :
    """
    derivative of the gradient of the total energy with respect to the gains when the optical pumping suffers from the same loss as the data transmission
    :param attenuation: a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :param gainList   : list of real numbers modelling amplifier gains [g[0], .... ] each larger than or equal to 1 of the evenly spaced amplifiers
    :param length     : positive real modelling total length of the fiber in [km]
    :ns               : non-negative real modelling the expected signal energy per pulse at the transmitter in photons
    """
    segments = len(gainList) + 1
    d = np.exp( - attenuation*length/segments )
    factor = [ ns*np.prod( [gainList[j] for j in range(i)] ) for i in range(segments-1)]
    term2 = [ sum([np.power(d,j+1)*(gainList[j] - 1) for j in range(i+1,segments-1)])/gainList[i]  for i in range(segments-1)]
    return [ np.power( 1/d, np.min([i+1, segments - i]))*factor[i]*(np.power(d,i+1) + term2[i]) for i in range(len(gainList))]        

def dGSH ( attenuation, gainList, i, length,  ns ):
    """
    derivative of the gradient of the Holevo capacity with respect to the gains
    :param attenuation: a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :param gainList   : list of real numbers modelling amplifier gains [g[0], .... ] each larger than or equal to 1 of the evenly spaced amplifiers
    :param length     : positive real modelling total length of the fiber in [km]
    :ns               : non-negative real modelling the expected signal energy per pulse at the transmitter in photons
    """
    segments = len(gainList) + 1
    d = np.exp( -attenuation*length/segments ) # be careful, this here is the transmittivity per segment
    dTotal = np.exp( -attenuation*length )
    totalNoise = vGainNoise( attenuation, gainList, length)
    partialGain = np.product([gainList[j] for j in range(i)])*np.product([gainList[j] for j in range(i+1,len(gainList))])
    v = np.array([0,1])
    for j in range(len(gainList)):
        if j != i:
            v = gainMatrix(gainList[j], d).dot(v)
        else:
            v = dGainMatrix( d ).dot(v)
    dNoise = v[0]
    term1 = dg(np.prod(gainList)*dTotal*ns +  d*totalNoise)*( partialGain*dTotal*ns + d*dNoise )
    term2 = dg(d*totalNoise)*d*dNoise
    return  term1 - term2

def dHom ( attenuation, gainList, i, length,  ns ):
    """
    derivative of the single-shot detector capacity when homodyne detection is used
    :param attenuation: a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :param gainList   : list of real numbers modelling amplifier gains [g[0], .... ] each larger than or equal to 1 of the evenly spaced amplifiers
    :i                : integer. The derivative is taken with respect to the i-th gain
    :param length     : positive real modelling total length of the fiber in [km]
    :ns               : non-negative real modelling the expected signal energy per pulse at the transmitter in photons
    """
    segments = len(gainList) + 1
    d = np.exp( -attenuation*length/segments ) # be careful, this here is the transmittivity per segment
    totalAtt = np.exp( -attenuation*length )
    totalNoise = vGainNoise( attenuation, gainList, length)
    totalGain = np.product(gainList)
    dGain = np.product([gainList[j] for j in range(i)])*np.product([gainList[j] for j in range(i+1,len(gainList))])
    v = np.array([0,1])
    for j in range(len(gainList)):
        if j != i:
            v = gainMatrix(gainList[j], d).dot(v)
        else:
            v = dGainMatrix( d ).dot(v)
    dNoise = v[0]
    S = sHomodyne(totalAtt * totalGain, ns, totalNoise)
    term1 = 4*dGain*totalAtt*ns/( 2 + totalNoise )
    term2 = -4*totalAtt*totalGain*ns*dNoise/np.power( 2 + totalNoise ,2)
    return ( term1 + term2 ) / S

def sholGradient( attenuation, gainList, length,  ns ):
    """
    gradient of the Holevo capacity with respect to the gains, evaluated using below parameters
    :param attenuation: a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :param gainList   : list of real numbers modelling amplifier gains [g[0], .... ] each larger than or equal to 1 of the evenly spaced amplifiers
    :param length     : positive real modelling total length of the fiber in [km]
    :ns               : non-negative real modelling the expected signal energy per pulse at the transmitter in photons
    """
    return [dGSH( attenuation, gainList, i, length,  ns ) for i in range(len(gainList))] 

def homodyneGradient( attenuation, gainList, length,  ns ):
    """
    gradient of the single-shot detector capacity with respect to the gains when homodyne detection is used, evaluated using below parameters
    :param attenuation: a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :param gainList   : a list of gains that are applied to the amplifiers which are assumed as evenly spaced
    :param length     : total length of the fiber in [km]
    :ns               : expected signal energy per pulse at the transmitter in photons (non-negative real)
    """
    return [dHom( attenuation, gainList, i, length,  ns ) for i in range(len(gainList))]

def maxGain( attenuation, length, ns, segments):
    """
    the maximum gain that is allowed under the assumption that the total number of signal plus noise photons should never exceed the initial number of noise photons
    :param attenuation: a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :param length     : positive real modelling total length of the fiber in [km]
    :ns               : non-negative real modelling the expected signal energy per pulse at the transmitter in photons
    :segments         : number of segments (segments -1 is the number of amplifiers)
    """
    d = np.exp( - attenuation*length/segments )
    """
    other return values are possible as well, for example we used the value  0.5*(1 + (ns + 1)/(d*ns + 1)) to study alternatives to the maximum gain
    """
    return (ns + 1)/(d*ns + 1)

def innerGradientOpt( benchmark, currentGs, maxG, accuracy, attenuation, length, ns, segments, gainlist, criterion, homodyne):
    """
    the inner loop of the optimization function follows the energy gradient eGradient (or expGradient), reducing gains
        -- until the Holevo capacity sHol reaches the value "benchmark".
    then it walks along the surface where sHol = benchmark to further minimize the energy
    :param benchmar   : a pre-defined positive real. The algorithm ensures that sHol never falls below this value.
    :currentGs        : a list of numbers g satisfying maxG >= g >= 1 which specifies the current gains
    :maxG             : the maximum allowed gain
    :accuracy         : this non-negative value specifies the maximum possible stepsize in the algorithm
    :param attenuation: a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :param length     : total length of the fiber in [km]
    :ns               : expected signal energy per pulse at the transmitter in photons (non-negative real)
    :segments         : number of segments (segments -1 is the number of amplifiers)
    :gainlist         : this list can be used to track gains
    :criterion        : choose a value "ePump", "expPum" or "rePump" to specify under which of these energy function to optimize
    :homodyne         : True or False - if set to True, it optimizes the single-shot detector capacity using homodyne detection instead of the Holevo capacity!!
    """
    stepsize = 0
    # this constant gradient can be used for REGS
    grad = [1/np.sqrt( segments - 1 ) for i in range( segments - 1 )]
    # now we move towards a surface where S = benchmark, thereby following the gradient of the objective function ePump, expPump or rePump (relaxation)
    if currentGs == [maxG for i in range(segments - 1)]:
        itlog = 0
        while stepsize < accuracy and itlog < 1000:
            itlog += 1
            if criterion == "ePump":
                eGrad = list(eGradient(attenuation, currentGs, length, ns ))
                norm = np.sqrt(np.dot(eGrad, eGrad))
                grad = [eGrad[i] / norm for i in range( segments -1)]
            if criterion == "expPump":
                eGrad = list(expGradient(attenuation, currentGs, length, ns ))
                norm = np.sqrt(np.dot(eGrad, eGrad))
                grad = [eGrad[i] / norm for i in range( segments -1)]
            step = (maxG - 1 )/(5*np.power(2, stepsize,dtype='int64'))
            newGs = [currentGs[x] - step * grad[x] for x in range( segments - 1)]
            condition1 = all([ maxG >= newGs[k] >= 1 for k in range( segments -1 )])
            condition2 = False
            if condition1:
                if homodyne:
                    condition2 = sHomodyne( np.exp( -attenuation*length)*np.prod( newGs ), ns, np.exp( -attenuation*length/segments)*vGainNoise( attenuation, newGs, length)) > benchmark
                else:
                    condition2 = sHol( np.exp( -attenuation*length)*np.prod( newGs ), ns, np.exp( -attenuation*length/segments)*vGainNoise( attenuation, newGs, length)) > benchmark
            if condition1 and condition2:
                currentGs = newGs[:]
                gainlist.append(currentGs)
            else:
                stepsize += 1
    else:
        stepsize = 0
        itlog = 0
        while stepsize < accuracy and itlog < 1000:
            itlog += 1
            step = (maxG - 1 )/(2*np.power(2, stepsize ,dtype='int64'))
            newGs = [( 1 + step)*currentGs[x] - step * maxG for x in range(segments - 1)]
            condition1 = all([ maxG >= newGs[k] >= 1 for k in range(len(newGs))])
            condition2 = False
            if condition1:
                if homodyne:
                    condition2 = sHomodyne( np.exp( -attenuation*length)*np.prod( newGs ), ns, np.exp( -attenuation*length/segments)*vGainNoise( attenuation, newGs, length)) > benchmark
                else:
                    condition2 = sHol( np.exp( -attenuation*length)*np.prod( newGs ), ns, np.exp( -attenuation*length/segments)*vGainNoise( attenuation, newGs, length)) > benchmark
            if condition1 and condition2:
                currentGs = newGs[:]
                gainlist.append(currentGs)
            else:
                stepsize += 1
    stepsize = 0
    itlog = 0
    # now we walk roughly on a surface where S is constant
    while stepsize < accuracy and itlog < 1000:
        itlog += 1 
        #calculate normalized gradient of ePump
        if criterion == "ePump":
            eGrad = list(eGradient(attenuation, currentGs, length, ns ))
            norm = np.sqrt(np.dot(eGrad, eGrad))
            grad = [eGrad[i] / norm for i in range( segments -1)]
        if criterion == "expPump":
            eGrad = list(expGradient(attenuation, currentGs, length, ns ))
            norm = np.sqrt(np.dot(eGrad, eGrad))
            grad = [eGrad[i] / norm for i in range( segments -1)]        #calculate normalized gradient of sHol
        if homodyne:
            sGrad = homodyneGradient(attenuation, currentGs, length, ns )
        else:
            sGrad = sholGradient(attenuation, currentGs, length, ns )
        sNorm = np.sqrt(np.dot(sGrad, sGrad))
        normedSGrad = [sGrad[i] / sNorm for i in range( segments -1 )]
        #calculate projection of grad onto orthocomplement of sGrad
        prod = np.dot(grad, normedSGrad)
        pGrad = [grad[i] - prod*normedSGrad[i] for i in range(segments -1)]
        pgNorm = np.sqrt(np.dot(pGrad,pGrad))
        if pgNorm > 0:
            projectedGrad = [ pGrad[i] / pgNorm for i in range( segments -1)]
        else:
            break
        step = (maxG - 1 )/(100*np.power(2, stepsize,dtype='int64'))
        newGs = [currentGs[x] - 1*step * projectedGrad[x] for x in range( segments - 1 )]
        condition1 = all([ maxG >= newGs[k] >= 1 for k in range(len(newGs))])
        condition2 = False
        if condition1:
            if homodyne:
                condition2 = sHomodyne( np.exp( -attenuation*length)*np.prod( newGs ), ns, np.exp( -attenuation*length/segments)*vGainNoise( attenuation, newGs, length)) > benchmark
            else:
                condition2 = sHol( np.exp( -attenuation*length)*np.prod( newGs ), ns, np.exp( -attenuation*length/segments)*vGainNoise( attenuation, newGs, length)) > benchmark
        if condition1 and condition2:
            currentGs = newGs[:]
            gainlist.append(currentGs)
        else:
            stepsize += 1
    return currentGs, gainlist

def gradientOpt( attenuation, length, ns, segments, criterion, homodyne, maxG ):
    """
    Calculates the best possible variable gains for the entire link. Starts with *maximum* gains.
    returns adjusted gains, achieved (quantum) spectral efficiency, and consumed pump energy.
    :param attenuation: a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :param length     : total length of the fiber in [km]
    :ns               : expected signal energy per pulse at the transmitter in photons (non-negative real)
    :segments         : number of segments (segments -1 is the number of amplifiers), the link is automatically divided into segments of equal length
    :criterion        : choose a value "ePump", "expPum" or "rePump" to specify under which of these energy function to optimize
    :homodyne         : True or False - if set to True, it optimizes the single-shot detector capacity using homodyne detection instead of the Holevo capacity!!
    :maxG             : the maximum allowed gain
    """
    printGainList = False
    d = np.exp( - attenuation*length/segments )    
    #maxG = maxGain( attenuation, length, ns, segments)
    maxGs = [ maxG for j in range( segments - 1 )]
    benchmark = sSha( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/segments)*vGainNoise( attenuation, maxGs, length))
    if benchmark == 0:
        return [1 for i in range( segments -1 )], sHol( np.exp( -attenuation*length), ns, 0), 0
    currentGs = maxGs[:]
    gainList = []
    #iterate the steps of following the energy gradient and then moving along a surface of constant S with increasing accuracy
    acc = 1
    while acc <= ACCURACY:
        currentGs, gainlist = innerGradientOpt( benchmark, currentGs, maxG, 6 + 2*acc, attenuation, length, ns, segments, gainList, criterion, homodyne)
        acc += 1
    if homodyne:
        S = sHomodyne( np.exp( -attenuation*length)*np.prod(currentGs), ns, d*vGainNoise( attenuation, currentGs, length))
    else:
        S = sHol( np.exp( -attenuation*length)*np.prod(currentGs), ns, d*vGainNoise( attenuation, currentGs, length))
    if criterion == "rePump":
        eP = rePump (attenuation, length, currentGs, ns)
    if criterion == "ePump":
        eP = ePump (attenuation, length, currentGs, ns)
    if criterion == "expPump":
        eP = expPump (attenuation, length, currentGs, ns)
    if printGainList:
        print("gradientOpt : gainlist = ",gainlist)
    return currentGs, S, eP
            
def linkEval( attenuation, length, ns, segments, crit, homodyne=False, amplify= "moderate" ):
    """
    evaluates whether attenuation makes sense, classically, and calculates the quantum savings if yes.
    :param attenuation: a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :param length     : total length of the fiber in [km]
    :ns               : expected signal energy per pulse at the transmitter in photons (non-negative real)
    :segments         : number of segments (segments -1 is the number of amplifiers), the link is automatically divided into segments of equal length
    :crit             : choose a value "ePump", "expPum" or "rePump" to specify under which of these energy function to optimize
    :homodyne         : True or False - if set to True, it optimizes the single-shot detector capacity using homodyne detection instead of the Holevo capacity!!
    :amplify          : if set to moderate, the maximum gain is chosen according to maxGain. Otherwise, it is simply set to np.exp( attenuation*length) to completely restore signal power
    """
    if amplify == "moderate":
        maxG = maxGain( attenuation, length, ns, segments)
    else:
        maxG = np.exp( attenuation*length)
    maxGs = [ maxG for j in range( segments - 1 )]
    benchmark = sSha( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/segments)*vGainNoise( attenuation, maxGs, length))
    lowerBound = sSha(np.exp( -attenuation*length), ns, 0)
    holPerf = 0
    if crit == "ePump":
        epSha = ePump( attenuation, length, maxGs, ns)
    if crit == "rePump":
        epSha = rePump( attenuation, length, maxGs, ns)
    if crit == "expPump":
        epSha = expPump( attenuation, length, maxGs, ns)
    percentage = 0
    if epSha > 0:
        gainList, S, eP = gradientOpt( attenuation, length, ns, segments, crit, homodyne, maxG )
        percentage = eP/epSha
        if homodyne:
            if S < benchmark:
                print("homodyne=",S," lower than Shannon=",benchmark)
                percentage = -1
    return percentage


def ampEval( attenuation, ns, maxSegments, targetMultipliers, start, stepSize, nSteps, homodyne ):
    """
    gives back a matrix M of steps where an entry M[t] displays where amplification gain of targetMultipliers[t] is reached
    :param attenuation: a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :ns               : expected signal energy per pulse at the transmitter in photons (non-negative real)
    :maxSegments      : integer signifying the maximum number of segments to consider
    :targetMultipliers: real number larger than 1, for example 2, 10 or 100.
                        For every k<maxSegments, the kilometer at which switching on k amplifiers yields an increase in Shannon capacity equal to a target multiplier is calculated. 
    :start            : smallest possible link length [km], must be larger than 0
    :stepSize         : stepsize used in optimization in [km]
    :nSteps           : integer. The value start + nSteps*stepSize is the maximum link length considered by this algorithm
    :homodyne         : True or False - if set to True, it optimizes the single-shot detector capacity using homodyne detection instead of the Holevo capacity!!
    """
    # gives back a matrix M of steps where an entry M[t] displays where amplification gain of targetMultipliers[t] is reached
    myMap = [[] for t in range(len(targetMultipliers))]
    for t in range(len(targetMultipliers)):
        for k in range(maxSegments):
            ratio = 1
            step = start
            while ratio < targetMultipliers[t] and step < 2*nSteps:
                length = start + step*stepSize
                maxG = maxGain( attenuation, length, ns, k+2)
                maxGs = [ maxG for j in range( k +1 )]
                if homodyne:
                    amplifiedS = sHomodyne( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/(k+2))*vGainNoise( attenuation, maxGs, length))
                    S = sHomodyne( np.exp( -attenuation*length), ns, 0)
                else:
                    amplifiedS = sSha( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/(k+2))*vGainNoise( attenuation, maxGs, length))
                    S = sSha( np.exp( -attenuation*length), ns, 0)
                ratio = np.inf
                if S > 0:
                    ratio = amplifiedS/S
                step += stepSize
            myMap[t].append([step, k + 1])
    return myMap


def specEval( attenuation, ns, maxSegments, maxSpecEff, start, stepSize, nSteps, homodyne ):
    """
    gives back a matrix M of steps where row M[s] is a list [km, amplifiers] containing kilomers and amplifier numbers at which the spectral efficiency with k amplifiers switched from being above s+1 to below s+1
    the last two rows of the matrix containt values for s=0.1 and s=0.01
    :param attenuation: a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :ns               : expected signal energy per pulse at the transmitter in photons (non-negative real)
    :maxSegments      : integer signifying the maximum number of segments to consider
    :maxSpecEff       : real number specifying the maximum spectral efficiency for which combinations (km, number of amplifiers) realizing exactly this spectral efficiency should be computed
    :start            : smallest possible link length [km], must be larger than 0
    :stepSize         : stepsize used in optimization in [km]
    :nSteps           : integer. The value start + nSteps*stepSize is the maximum link length considered by this algorithm
    :homodyne         : True or False - if set to True, it optimizes the single-shot detector capacity using homodyne detection instead of the Holevo capacity!!
    """
    myMap = [[] for s in range(maxSpecEff + 2)]
    minimum = maxSpecEff
    for s in range(maxSpecEff):
        for k in range(maxSegments):
            length = start/10
            maxG = maxGain( attenuation, length, ns, k+1)
            maxGs = [ maxG for j in range( k )]
            if homodyne:
                S = sHomodyne( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/(k+1))*vGainNoise( attenuation, maxGs, length))
            else:
                S = sSha( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/(k+1))*vGainNoise( attenuation, maxGs, length))
            new = 0
            step = 0
            while S > s + 1 and step < 2*nSteps:
                length = start + step*stepSize
                maxG = maxGain( attenuation, length, ns, k+1)
                maxGs = [ maxG for j in range( k )]
                if homodyne:
                    S = sHomodyne( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/(k+1))*vGainNoise( attenuation, maxGs, length))
                else:
                    S = sSha( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/(k+1))*vGainNoise( attenuation, maxGs, length))
                step += stepSize
            myMap[s].append([step, k + 1])
    if True:
        for k in range(maxSegments):
            length = start/10
            maxG = maxGain( attenuation, length, ns, k+1)
            maxGs = [ maxG for j in range( k )]
            if homodyne:
                S = sHomodyne( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/(k+1))*vGainNoise( attenuation, maxGs, length))
            else: 
                S = sSha( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/(k+1))*vGainNoise( attenuation, maxGs, length))
            if S < minimum:
                minimum = S
            new = 0
            step = 0
            while S > 0.1 and step < 2*nSteps:
                length = start + step*stepSize
                maxG = maxGain( attenuation, length, ns, k+1)
                maxGs = [ maxG for j in range( k )]
                if homodyne:
                    S = sHomodyne( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/(k+1))*vGainNoise( attenuation, maxGs, length))
                else:
                    S = sSha( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/(k+1))*vGainNoise( attenuation, maxGs, length))
                step += stepSize
            myMap[maxSpecEff].append([step, k + 1 ])
    if True:
        for k in range(maxSegments):
            length = start/10
            maxG = maxGain( attenuation, length, ns, k+1)
            maxGs = [ maxG for j in range( k )]
            if homodyne:
                S = sHomodyne( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/(k+1))*vGainNoise( attenuation, maxGs, length))
            else:
                S = sSha( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/(k+1))*vGainNoise( attenuation, maxGs, length))
            if S < minimum:
                minimum = S
            new = 0
            step = 0
            while S > 0.01 and step < 2*nSteps:
                length = start + step*stepSize
                maxG = maxGain( attenuation, length, ns, k+1)
                maxGs = [ maxG for j in range( k )]
                if homodyne:
                    S = sHomodyne( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/(k+1))*vGainNoise( attenuation, maxGs, length))
                else:
                    S = sSha( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/(k+1))*vGainNoise( attenuation, maxGs, length))
                step += stepSize
            myMap[maxSpecEff + 1].append([step, k + 1 ])
    return myMap

def createMapData( attenuation, length, ns, maxSegments, start, stepSize, nSteps, criterion, homodyne=False ):
    """
    umbrella function to execute any of the tasks specified by <criterion>.
    :param attenuation: a non-negative number that models the transmittivity of a link of length L as exp(-attenuation * L)
    :ns               : expected signal energy per pulse at the transmitter in photons (non-negative real)
    :maxSegments      : integer signifying the maximum number of segments to consider
    :start            : smallest possible link length [km], must be larger than 0
    :stepSize         : stepsize used in optimization in [km]
    :nSteps           : integer. The value start + nSteps*stepSize is the maximum link length considered by this algorithm
    :criterion        : can be any of ["ePump", "rePump", "expPump"] to calculate energy savings.
                            If set to "specLines" it calculates pairs (km, number of amplifiers) at which the spectral efficiency is constant
                            If set to "ampLines" it calculates pairs (km, number of amplifiers) at which the Shannon spectral efficiency jumps by factors of [1.1, 2, 10, 100] if the amplifiers are set to the maximum gain.
    :homodyne         : True or False - if set to True, it optimizes the single-shot detector capacity using homodyne detection instead of the Holevo capacity!!
    """
    end = start + nSteps * stepSize
    nCores = int(multiprocessing.cpu_count())
    pool = Pool(nCores)
    print("using",nCores,"CPUs")
    if criterion in ["ePump", "rePump", "expPump"]:
        myMap = [[0 for k in range(int(np.ceil(nSteps/nCores)*nCores))] for i in range(maxSegments)]
    if criterion == "specLines":
        print("calculating lines along which the spectral efficiency is constant with homodyne set to ",homodyne)
        return specEval( attenuation, ns, maxSegments, 20, start, stepSize, nSteps, homodyne)
    if criterion == "ampLines":
        print("calculating lines along which the gain in spectral efficiency resulting from amplification is constant with homodyne set to ",homodyne)
        return ampEval( attenuation, ns, maxSegments, [1.1,2,10,100], start, stepSize, nSteps, homodyne)
    for k in range(maxSegments):
        # get an idea of the time it takes to compute here
        startTime = int(time.time() )
        print("number of amplifiers is ",k,"start time of iteration is ",datetime.now()," criterion=",criterion," homodyne is set to ",homodyne)
        def otf(i, res, x):
            res[x] = linkEval()
        if k > 1:
            for i in range(0,int(np.ceil(nSteps/nCores))):
                results = pool.starmap(linkEval, [(attenuation, start + i*nCores*stepSize + j, ns, k+1, criterion, homodyne) for j in range(nCores)])
                for j in range(int(nCores)):
                    myMap[k][i*nCores + j] = results[j]
                if i%nCores == 1:
                    print(round(100*i*nCores/nSteps),"% finished in round ",k)
        else:
            for i in range(nSteps):
                res = linkEval(attenuation, start + i*stepSize, ns, k+1, criterion, homodyne)
                myMap[k][i] = res
                if i%int(nSteps/20) == 1:
                    print(round(100*i/nSteps),"% finished in round",k)

        a = np.asarray(myMap)
        if criterion == "rePump":
            path = "v11-" + str(start) + "-" + str(end) + "att=" + str(attenuation) + "hom=" + str(homodyne) + "rePump-n_S=" + str(ns) + ".csv"
        else:
            path = "v11-" + str(start) + "-" + str(end) + "att=" + str(attenuation) + "hom=" + str(homodyne) + "ePump-n_S=" + str(ns) + ".csv"
        #np.savetxt(path, a, delimiter=",")
        endTime = int(time.time() )
        print("this round took ", endTime - startTime, " seconds" )
    return myMap


def createCsvData(st=1, nSt=1000, stS=1, crit="ePump", nPhotons=np.power(10,4),attenuation=0.05, hom=False, maxNumberOfAmplifiers=10, path=""):
    """
    saves a CSV file containing triples (km, number of amplifiers, energy savings through joint detection)
    :st                   : positive real defining the shortest link length [km] 
    :nSt                  : integer. The value st + nSt*stS is the maximum link length considered by this algorithm
    :stS                  : stepsize used in optimization in [km]
    :crit                 : can be any of ["ePump", "rePump", "expPump"] to calculate energy savings.
    :nPhotons             : positive real : expected signal energy per pulse at the transmitter in photons (non-negative real)
    :attenuation          : positive real : the transmittivity is calculated as exp( - attenuation*length) where length is the linklength
    :hom                  : True or False - if set to True, it optimizes the single-shot detector capacity using homodyne detection instead of the Holevo capacity!!
    :maxNumberOfAmplifiers: integer signifying the maximum number of segments to consider
    :path                 : string which is used in the specificiation of the CSV file name
    """
    ed = st + nSt*stS
    currentTime = int(time.time())
    path = str(currentTime) + path
    if crit in ["ampLines", "ePump","rePump", "expPump", "specLines"]:
        out = createMapData( attenuation, ed, nPhotons, maxNumberOfAmplifiers, st, stS, nSt, crit, hom )
        a = np.asarray(out)
        if crit == "ampLines":
            path += "-" + str(st) + "-" + str(ed) + "att=" + str(attenuation) + "hom=" + str(hom) + "ampLines=[" + str(1.1) + "," + str(2) + "," + str(10) + "," + str(100) + "-n_S=" + str(nPhotons) + ".csv"
        if crit == "specLines":
            path += "-" + str(st) + "-" + str(ed) + "att=" + str(attenuation) + "hom=" + str(hom) + "specLines=" + "[1,...,20,0.01,0.1]" + "-n_S=" + str(nPhotons) + ".csv"
        if crit == "rePump":
            path += "-" + str(st) + "-" + str(ed) + "att=" + str(attenuation) + "hom=" + str(hom) + "rePump-n_S=" + str(nPhotons) + ".csv"
        if crit == "expPump":
            path += "-" + str(st) + "-" + str(ed) + "att=" + str(attenuation) + "hom=" + str(hom) + "expPump-n_S=" + str(nPhotons) + ".csv"
        if crit == "ePump":
            path += "-" + str(st) + "-" + str(ed) + "att=" + str(attenuation) + "hom=" + str(hom) + "expPump-n_S=" + str(nPhotons) + ".csv"
        np.savetxt(path, a, delimiter=",")
        if crit in ["ePump", "rePump", "expPump"]:
            plt.rcParams["figure.figsize"] = [7.50, 3.50]
            plt.rcParams["figure.autolayout"] = True
            plt.imshow(out, cmap='hot', interpolation='nearest')
        else:
            for j in range(len(out)):
                print(out[j])
                plt.plot([out[j][i][0] for i in range(len(out[j]))], [out[j][i][1] for i in range(len(out[j]))])
        plt.show()
        
def generateData(alph, LL):
    """
    compares savings at distance set in LL
    :alph: attenuation coefficient
    :LL  : length in [km] - transmittivity is exp( -0.05*LL)
    Regarding the number of photons per pulse "nP" defined below :
    At 1550nm, a 100mW transmitter will emit at most 100*10^16 photons per second. Current systems use no more than 100mW.
    These 100mW are spread over all frequencies used by the different channels. According to ITU there are 144 50GHz channels in the C-band.
    Today's commercial systems emit in the order of 10*10^9 pulses per second, so that nP=10^5 reflects today's systems.
    A very futuristic system might have 100 times more pulses per second, using both C- and O band, for example. Then setting nP=10^3 is more accurate.
    """
    scale = 1
    br = scale * 50 * (10**9)
    carriers = 0.000001*1.3888*144/scale
    # we include the time-bandwidth product with a value of 0.4 when calculating the bandwidth. A value of 0.25 is the theoretical minimum.
    bandwidth = 0.4*br*carriers
    bandwidthOfCBand = 4.4*(10**12)
    print("bandwidth is",bandwidth/(10**12),"THz, corresponding to ",int(100*bandwidth/bandwidthOfCBand),"% of the C-Band")
    nP = ( 10**(16 - math.log(br, 10)) )/carriers
    dat = []
    for F in range(9):
        crs = np.power(10,F)*2
        nP = ( 10**(20 - math.log(br, 10)) )/crs
        print("nP=10^",math.log(nP,10))
        mins = []
        avgs = []
        stds = []
        for s in range(9):
            print("    # amplifiers=",s+1)
            sParams = []
            minL=np.inf
            minR=np.inf
            for l in range(11):
                sgs = s+2
                L = LL - 1 + l*0.2
                lev = linkEval( alph, L, nP, sgs, "ePump", False, "strong" )
                sParams.append( lev )
                if l<6:
                    if lev < minL:
                        minL = lev
                else:
                    if lev < minR:
                        minR = lev                    
            avg = 0
            for sp in sParams:
                avg += sp
            avg = avg/11
            std = 0
            for sp in sParams:
                std += (sp - avg)**2
            std = np.sqrt(std/11)
            avgs.append(avg)
            stds.append(std)
            print(minL,minR,minL/minR," -- ",avg,avg/minL)
            maxmin = max([minL,minR])
            mins.append(maxmin)
        
        dat.append([avgs,stds,mins])     
        
    print(dat)

def mWtoPhotonN(mw, wavelength, baudrate):
    """
    calculate number of photons per pulse from transmitter energy in [mW], center wavelength in [nm], and baud-rate in pulses per second
    exponent is calculated as -3 + 34 - 8 - 9 with exponents coming from mW scale, Planck constant, speed of light in fiber, nm scale
    : mw        : power at transmitter in milli Watt
    : wavelength: the center wavelength in nanometer
    : baudrate  : the number of pulses per second. Today, 10^10 is a reasonable value.
    """
    return mw*np.power(10, -20)/(baudrate*6.625*2/mw)

if __name__ == '__main__':
    # Test the algorithm with some numbers, like so:
    #createCsvData(1, 1000, 1, "rePump", np.power(10,3), 0.05, False, 10, "n=pwr(10,3)-1000km-1-9amps-EGS")
    alph = 0.05
    L = 164
    # number of photons per pulse. At 1550nm, a 100mW transmitter will emit at most 100*10^16 photons per second. Current systems use no more than 100mW.
    # Today's commercial systems emit 100*10^9 pulses per second, so that nP=10^7 reflects today's systems.
    # A futuristic system might have 100 times more pulses per second, using both C- and O band, for example. Then setting nP=10^5 is more accurate.
    scale = 1
    br = scale * 50 * (10**9)
    carriers = 1*1.3888*144/scale
    # we include the time-bandwidth product with a value of 0.4 when calculating the bandwidth. A value of 0.25 is the theoretical minimum.
    bandwidth = 0.4*br*carriers
    bandwidthOfCBand = 4.4*(10**12)
    print("baud-rate is",br/(10**9),"GBaud")
    print("bandwidth is",bandwidth/(10**12),"THz, corresponding to ",int(100*bandwidth/bandwidthOfCBand),"% of the C-Band")
    nP = ( 10**(18 - math.log(br, 10)) )/carriers
    sgs = 5
    print("photon number per pulse is 10^",math.log(nP,10))
    maxG = maxGain( alph, L, nP, sgs)
    maxG = np.exp(alph*L/sgs)
    maxGs = [ maxG for j in range( sgs - 1 )]
    benchmark = sSha( np.exp( -alph*L)*np.prod( maxGs ), nP, np.exp( -alph*L/sgs)*vGainNoise( alph, maxGs, L))
    print("total noise at the receiver is calculated to be",np.exp( -alph*L/sgs)*vGainNoise( alph, maxGs, L))
    print("cross-check with Mathematica ", br*carriers*sHol( np.exp( -alph*L)*maxG, nP, np.exp( -alph*L/sgs)*(10.3056 - 1 ))/(10**12),"TBit/s")
    quantumBenchmark = sHol( np.exp( -alph*L)*np.prod( maxGs ), nP, np.exp( -alph*L/sgs)*vGainNoise( alph, maxGs, L))
    lowerBound = sSha(np.exp( -alph*L), nP, 0)
    print("photon number directly before the first amplifier is",                       nP*np.exp(-alph*L/sgs)      )
    print("amplified current system reaches S=",                                        benchmark                   ) 
    print("    corresponding capacity [Tera bit/s] :",                                  carriers*benchmark*br/(10**12)       )
    print("    replacing the receiver with a JDR yields a capacity of :",               carriers*quantumBenchmark*br/(10**12),"[Tera bit/s]"     )
    print("    replacing the receiver with a JDR yields a SE of :",               quantumBenchmark   )
    print("maxGain=",maxG               )
    print("non-amplified Shannon Capacity [Tera bit/s] =",carriers*br*lowerBound/(10**12))
    print("non-amplified Shannon SE [Tera bit/sxHz] =",lowerBound/(10**12))
    print("amplification yields an improvement of AE=",benchmark / lowerBound)
    print("non-amplified Holevo Capacity [Tera bit/s] =",carriers*br*sHol(np.exp( -alph*L), nP, 0)/(10**12))
    print("non-amplified Holevo SE [Tera bit/sxHz] =",sHol(np.exp( -alph*L), nP, 0)/(10**12))
    # increase accuracy in the calculation with the global variable ACCURACY #
    #print("if gains on transmission line with JDR is adjusted to only reach the same capacity as the OSSR at maximum gain it uses",linkEval( alph, L, nP, sgs, "ePump", False ),"percent of the energy")
    #createCsvData(st=1, nSt=999, stS=1, crit="expPump", nPhotons=np.power(10,5),attenuation=0.05, hom=False, maxNumberOfAmplifiers=10)
    #  after generating the raw data, you might want to know how to get the lines where S or AE are constant. You need to run
    #print(specEval( attenuation=0.05, ns=np.power(10,5), maxSegments=10, maxSpecEff=14, start=1, stepSize=1, nSteps=999, homodyne=False ))
    #  run ampEval to learn where amplifiers yield an increase
    #print(ampEval( attenuation=0.05, ns=np.power(10,5), maxSegments=10, targetMultipliers=[1.1,2,10,100], start=1, stepSize=1, nSteps=999, homodyne=False ))
    # get an overview of what happens at link lengths which are multiples of 80km :
    for i in range(10):
        print("energy percentage saved with",i,"amplifiers at ",(i+1)*80,"kilometers = ",1-linkEval( 0.05, (i+1)*80, np.power(10,5), i+1, "expPump"))

