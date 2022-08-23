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


ACCURACY = 50


# first we define the channel models 

def g (x):
    if x>0:
        return np.log2( 1 + x) + x*np.log2(1 + 1/x)
    else:
        return np.log2( 1 + x)

def dg (x):
    if x>0:
        return np.log2( 1 + 1/x)
    else:
        return np.power(10,10) # well, that's not really infinity but should suffice

# this is the spectral efficiency according to the Holevo formula
def sHol (tau, ns, noise):
    return g( ns*tau + noise ) - g( noise)

# this is the spectral efficiency according to the Shannon Hartley formula    
def sSha (tau, ns, noise):
    return np.log2( 1 + tau*ns/( 1 + noise ) )

# this is the spectral efficiency using homodyne Detection    
def sHomodyne (tau, ns, noise):
    return np.log2( 1 + 4*tau*ns/( 1 + 2*noise ) )/2

# now we define the noise model
def vGainNoise ( attenuation, gainList, length ) :
    noiseval = 0
    segments = len(gainList) + 1
    d = np.exp( -attenuation*length/segments )#be careful, this here is the transmittivity per segment
    for i in range(len(gainList)):
        if gainList[i] < 1: # make sure all gains stay above 1, otherwise throw exception
            raise Exception("gain number ",i," is smaller than 1")
        noiseval = gainList[i]*d*noiseval + gainList[i] - 1
    return noiseval

def ePump (attenuation, length, gainList, ns) :
    # computes the energy cost associated to a link with given parameters, up to a constant factor
    segments = len(gainList) + 1
    d = np.exp( -attenuation*length/segments )    
    noises = [0]
    taus = [1]
    lengths = [i*length/segments for i in range(len(gainList)+1)]
    for i in range(len(gainList)):
        noises.append(vGainNoise(attenuation, [ gainList[j] for j in range(i+1)], length*(i+2)/segments ) )
        taus.append(np.power(d,i+1))
    amplifiedTaus = [taus[i]*np.prod([ gainList[j] for j in range(i)] ) for i in range(len(gainList)+1)]
    return sum( [  ( gainList[i] - 1 )*( d*( amplifiedTaus[i]*ns + noises[i]) + 1 ) for i in range(len(gainList)) ] )

def rePump (attenuation, length, gainList, ns) :
    # computes the energy cost associated to a link with given parameters, up to a constant factor
    segments = len(gainList) + 1
    d = np.exp( -attenuation*length/segments )    
    return sum( [  ( gainList[i] - 1 )*( d*ns + 1 ) for i in range(len(gainList)) ] )


def gainMatrix( gain, tau ):
    return np.array([[ tau*gain, gain -1 ], [0, 1]])

def dGainMatrix( tau ):
    return np.array([[ tau, 1 ], [0, 0]])

def nuGradients (attenuation, gainList, length, ns, segments) :
    grad = [[0 for i in range(len(gainList))] for j in range(len(gainList))]
    d = np.exp( - attenuation*length/segments )
    for k in range(len(gainList)):
        for i in range( k + 1 ):
            v = np.array([0,1])
            for j in range( k + 1 ):
                if j != i:
                    v = gainMatrix(gainList[j], d).dot(v)
                else:
                    v = dGainMatrix( d ).dot(v)
            grad[k][i] = v[0]
    return grad

def tauGradients (attenuation, gainList, length, ns, segments) :
    grad = [[0 for i in range(len(gainList))] for j in range(len(gainList))]
    d = np.exp( - attenuation*length/segments )
    for k in range(len(gainList)):
        for i in range( k + 1 ):
            grad[k][i] = np.power(d, k)*np.product([gainList[j] for j in range(i)])*np.product([gainList[j] for j in range(i+1,k+1 )])
    return grad

def eGradient (attenuation, gainList, length, ns, segments) :
    grad = [0 for i in range(len(gainList))] 
    nuGrad = nuGradients(attenuation, gainList, length, ns, segments)
    tauGrad = tauGradients(attenuation, gainList, length, ns, segments)
    d = np.exp( - attenuation*length/segments )
    for i in range(len(gainList)):
        term1 = 1 + d*(np.power(d,i-1)*np.product([gainList[j] for j in range(i)])*ns + vGainNoise( attenuation, [gainList[j] for j in range(i)], length))
        term2 = d*sum([ (gainList[j] -1)*(tauGrad[j-1][i]*ns + nuGrad[j-1][i]) for j in range(i+1,len(gainList) ) ] )
        grad[i] = term1 + term2
    return grad


def dGSH ( attenuation, gainList, i, length,  ns ):
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
    return [dGSH( attenuation, gainList, i, length,  ns ) for i in range(len(gainList))] 

def homodyneGradient( attenuation, gainList, length,  ns ):
    return [dHom( attenuation, gainList, i, length,  ns ) for i in range(len(gainList))]

def maxGain( attenuation, length, ns, segments):
    d = np.exp( - attenuation*length/segments ) 
    return (ns + 1)/(d*ns + 1)

def innerGradientOpt( benchmark, currentGs, maxG, accuracy, attenuation, length, ns, segments, gainlist, relaxation, homodyne):
    stepsize = 0
    # this constant gradient can be used for ROGS
    grad = [1/np.sqrt( segments - 1 ) for i in range( segments - 1 )]
    # now we move towards a surface where S = benchmark, thereby following the gradient of the objective function in OGS or for ROGS
    if currentGs == [maxG for i in range(segments - 1)]:
        itlog = 0
        while stepsize < accuracy and itlog < 1000:
            itlog += 1
            if not relaxation:
                eGrad = list(eGradient(attenuation, currentGs, length, ns, segments))
                norm = np.sqrt(np.dot(eGrad, eGrad))
                grad = [eGrad[i] / norm for i in range( segments -1)]
            step = (maxG - 1 )/(5*np.power(2, stepsize))
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
            step = (maxG - 1 )/(2*np.power(2, stepsize))
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
        if not relaxation:
            eGrad = list(eGradient(attenuation, currentGs, length, ns, segments))
            norm = np.sqrt(np.dot(eGrad, eGrad))
            grad = [eGrad[i] / norm for i in range( segments -1)]
        #calculate normalized gradient of sHol
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
        step = (maxG - 1 )/(100*np.power(2, stepsize))
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

def gradientOpt( attenuation, length, ns, segments, relaxation, homodyne ):
    # Calculates the best possible variable gains for the entire link. Starts with *maximum* gains.
    # Returns adjusted gains, achieved (quantum) spectral efficiency, and consumed pump energy.
    printGainList = False
    d = np.exp( - attenuation*length/segments )    
    maxG = maxGain( attenuation, length, ns, segments)
    maxGs = [ maxG for j in range( segments - 1 )]
    benchmark = sSha( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/segments)*vGainNoise( attenuation, maxGs, length))
    if benchmark == 0:
        return [1 for i in range( segments -1 )], sHol( np.exp( -attenuation*length), ns, 0), 0
    currentGs = maxGs[:]
    gainList = []
    #iterate the steps of following the energy gradient and then moving along a surface of constant S with increasing accuracy
    acc = 1
    while acc <= ACCURACY:
        currentGs, gainlist = innerGradientOpt( benchmark, currentGs, maxG, 6 + 2*acc, attenuation, length, ns, segments, gainList, relaxation, homodyne)
        acc += 1
    if homodyne:
        S = sHomodyne( np.exp( -attenuation*length)*np.prod(currentGs), ns, d*vGainNoise( attenuation, currentGs, length))
    else:
        S = sHol( np.exp( -attenuation*length)*np.prod(currentGs), ns, d*vGainNoise( attenuation, currentGs, length))
    if relaxation:
        eP = rePump (attenuation, length, currentGs, ns)
    else:
        eP = ePump (attenuation, length, currentGs, ns)
    if printGainList:
        print("gradientOpt : gainlist = ",gainlist)
    return currentGs, S, eP
            
def linkEval( attenuation, length, ns, segments, crit, homodyne=False ):
    # evaluates whether attenuation makes sense, classically, and calculates the quantum savings if yes.
    maxG = maxGain( attenuation, length, ns, segments)
    maxGs = [ maxG for j in range( segments - 1 )]
    benchmark = sSha( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/segments)*vGainNoise( attenuation, maxGs, length))
    lowerBound = sSha(np.exp( -attenuation*length), ns, 0)
    holPerf = 0
    epSha = ePump( attenuation, length, maxGs, ns)
    percentage = 0
    if epSha > 0:
        if crit == "ePump":
            gainList, S, eP = gradientOpt( attenuation, length, ns, segments, False, homodyne )
            percentage = eP/epSha
        if crit == "rePump":
            gainList, S, eP = gradientOpt( attenuation, length, ns, segments, True, homodyne )
            percentage = eP/epSha
        if homodyne:
            if S < benchmark:
                print("homodyne=",S," lower than Shannon=",benchmark)
                percentage = -1
    return percentage


def ampEval( attenuation, ns, maxSegments, targetMultipliers, start, stepSize, nSteps, homodyne ):
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
    # gives back a matrix M of steps where row M[s] is a list [km, amplifiers] containing kilomers and amplifier numbers at which the spectral efficiency with k amplifiers switched from being above s+1 to below s+1
    # the last two rows of the matrix containt values for s=0.1 and s=0.01
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
    end = start + nSteps * stepSize
    if criterion == "ePump" or "rePump":
        myMap = [[0 for k in range(nSteps)] for i in range(maxSegments)]
    if criterion == "specLines":
        print("calculating lines along which the spectral efficiency is constant with homodyne set to ",homodyne)
        return specEval( attenuation, ns, maxSegments, 20, start, stepSize, nSteps, homodyne)
    if criterion == "ampLines":
        print("calculating lines along which the gain in spectral efficiency resulting from amplification is constant with homodyne set to ",homodyne)
        return ampEval( attenuation, ns, maxSegments, [1.1,2,10,100], start, stepSize, nSteps, homodyne)
    for k in range(maxSegments):
        # get an idea of the time it takes to compute here
        startTime = int(time.time() )
        print("number of amplifiers is ",k,"start time of iteration is ",datetime.now()," homodyne is set to ",homodyne)
        for i in range(0,nSteps):
            length = start + i*stepSize
            if i%20 == 1:
                print(round(100*i/nSteps),"% finished in round ",k)
            if i == nSteps - 1:
                maxG = maxGain( attenuation, length, ns, k+1)
                maxGs = [ maxG for j in range( k )]
                benchmark = sSha( np.exp( -attenuation*length)*np.prod( maxGs ), ns, np.exp( -attenuation*length/(k+1))*vGainNoise( attenuation, maxGs, length))
                print("    benchmark is",benchmark)           
            if criterion == "ePump" or "rePump":
                res = linkEval( attenuation, length, ns, k+1, crit, homodyne)
                myMap[k][i] = res
        a = np.asarray(myMap)
        if criterion == "rePump":
            path = "v11-" + str(st) + "-" + str(ed) + "att=" + str(attenuation) + "hom=" + str(hom) + "rePump-n_S=" + str(nPhotons) + ".csv"
        else:
            path = "v11-" + str(st) + "-" + str(ed) + "att=" + str(attenuation) + "hom=" + str(hom) + "ePump-n_S=" + str(nPhotons) + ".csv"
        np.savetxt(path, a, delimiter=",")
        endTime = int(time.time() )
        print("this round took ", endTime - startTime, " seconds" )
    return myMap

def createCsvData(st=1, nSt=1000, stS=1,crit="ePump",nPhotons=np.power(10,4),attenuation=0.05, hom=False,maxNumberOfAmplifiers=10, path="test"):
    if crit in ["ampLines", "ePump","rePump", "specLines"]:
        out = createMapData( attenuation, ed, nPhotons, maxNumberOfAmplifiers, st, stS, nSt, crit, hom )
        a = np.asarray(out)
        if crit == "ampLines":
            path += "-" + str(st) + "-" + str(ed) + "att=" + str(attenuation) + "hom=" + str(hom) + "ampLines=[" + str(1.1) + "," + str(2) + "," + str(10) + "," + str(100) + "-n_S=" + str(nPhotons) + ".csv"
        if crit == "specLines":
            path += "-" + str(st) + "-" + str(ed) + "att=" + str(attenuation) + "hom=" + str(hom) + "specLines=" + "[1,...,20,0.01,0.1]" + "-n_S=" + str(nPhotons) + ".csv"
        if crit == "rePump":
            path += "-" + str(st) + "-" + str(ed) + "att=" + str(attenuation) + "hom=" + str(hom) + "rePump-n_S=" + str(nPhotons) + ".csv"
        if crit == "rePump" or crit == "ePump":
            np.savetxt(path, a, delimiter=",")
        else:
            path += "-" + str(st) + "-" + str(ed) + "att=" + str(attenuation) + "hom=" + str(hom) + "ePump-n_S=" + str(nPhotons) + ".csv"
            with open(path, 'w') as outfile:
                for i in range(len(out)):
                    text = crit + " line " + str(i)
                    outfile.write('# ' + text + '\n')
                    np.savetxt(outfile, np.array(out[i]), fmt = '%-7.2f')
        #now display the results:
        if crit == "ePump" or crit == "rePump":
            plt.imshow(out, cmap='hot', interpolation='nearest')
        else:
            for j in range(len(out)):
                print(out[j])
                plt.plot([out[j][i][0] for i in range(len(out[j]))], [out[j][i][1] for i in range(len(out[j]))])
        plt.show()

if __name__ == '__main__':
    # Test the algorithm with some numbers, like so:
    alph = 0.05
    L = 600
    # number of photons per pulse. At 1550nm, a 100mW transmitter will emit at most 100*10^16 photons per second. Current systems use no more than 100mW.
    # Today's commercial systems emit 100*10^9 pulses per second, so that nP=10^7 reflects today's systems.
    # A futuristic system might have 100 times more pulses per second, using both C- and O band, for example. Then setting nP=10^5 is more accurate.
    br = 10**15
    nP = 10**(18 - math.log(br, 10))
    sgs = 6
    maxG = maxGain( alph, L, nP, sgs)
    maxGs = [ maxG for j in range( sgs - 1 )]
    benchmark = sSha( np.exp( -alph*L)*np.prod( maxGs ), nP, np.exp( -alph*L/sgs)*vGainNoise( alph, maxGs, L))
    quantumBenchmark = sHol( np.exp( -alph*L)*np.prod( maxGs ), nP, np.exp( -alph*L/sgs)*vGainNoise( alph, maxGs, L))
    lowerBound = sSha(np.exp( -alph*L), nP, 0)
    print("photon number per pulse is",nP)
    print("photon number at first amplifier is",nP*np.exp(-alph*L/sgs))
    print("amplified current system reaches S=",benchmark)
    print("    corresponding capacity [Tera bit/s] :",benchmark*br/(10**12))
    print("    replacing the receiver with a JDR yields a capacity of ",quantumBenchmark*br/(10**12),"[Tera bit/s]")
    print("maxGain=",maxG)
    print("non-amplified Shannon SE=",lowerBound)
    print("amplification yields an improvement of AE=",benchmark / lowerBound)
    print("non-amplified Holevo SE=",sHol(np.exp( -alph*L), nP, 0))
    # increase accuracy in the calculation with the global variable ACCURACY #
    print("quantum transmission line uses",linkEval( alph, L, nP, sgs, "ePump", False ),"percent of the energy")
