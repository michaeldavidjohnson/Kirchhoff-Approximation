from scipy.integrate import quad,romberg
from scipy.misc import derivative
from sklearn.metrics import mean_squared_error
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import clear_output
from numpy import matlib as mb
import csv
import time

class Directed2DVectorised:
    def __init__(self,sourceLocation,receiverLocations,surfaceFunction,frequency,a,sourceAngle = -np.pi/4,method = 'trapz',userMinMax = None,userSamples = 9000,absolute = True):
        self.sourceLocation = np.array(sourceLocation)
        self.receiverLocationsX = np.array(receiverLocations)[:,0]
        self.receiverLocationsY = np.array(receiverLocations)[:,1]
        self.k = (2*np.pi*frequency)/343
        self.a = a
        self.sourceAngle = sourceAngle
        self.surfaceFunction = surfaceFunction
        self.method = method
        self.absolute = absolute
        if userMinMax == None:
            self.min = self.receiverLocationsX.min()
            self.max = self.receiverLocationsX.max()
        else:
            self.min = - userMinMax
            #self.min = 0
            self.max = userMinMax
        self.number = self.receiverLocationsX.shape[0]
        self.samples = userSamples
        self.receiverLocationsX = self.receiverLocationsX.reshape(-1,1) + np.zeros((self.number,self.samples))
        self.receiverLocationsY = self.receiverLocationsY.reshape(-1,1) + np.zeros((self.number,self.samples))
        self.sourceLocationX = self.sourceLocation[0] + np.zeros((self.number,self.samples))
        self.sourceLocationY = self.sourceLocation[1] + np.zeros((self.number,self.samples))
        #BUG[IGGY] - FIX this, something is breaking and I'm not really sure why
        if type(self.surfaceFunction) == np.ndarray:
          self.surfaceVals = self.surfaceFunction 
          self.derivativeVals = np.diff(self.surfaceVals)/np.diff(x)
          self.doubleDerivativeVals = np.diff(self.derivativeVals)/np.diff(x)

          self.surfaceVals = self.surfaceVals.reshape(1,-1) + np.zeros((self.number,self.samples))
          self.derivativeVals = self.derivativeVals.reshape(1,-1) + np.zeros((self.number,self.samples))
          self.doubleDerivativeVals = self.doubleDerivativeVals.reshape(1,-1) + np.zeros((self.number,self.samples))
          self.x = np.linspace(self.min,self.max,self.samples).reshape(1,-1) + np.zeros((self.number,self.samples))

        else:
          
          self.x = np.linspace(self.min,self.max,self.samples).reshape(1,-1) + np.zeros((self.number,self.samples)) #see if this can be changed
          self.surfaceVals = surfaceFunction(self.x)
          self.derivativeVals = np.gradient(self.surfaceVals[0], self.x[0][1] - self.x[0][0],edge_order=2, axis=None)
          self.doubleDerivativeVals = np.gradient(self.derivativeVals, self.x[0][1] - self.x[0][0],edge_order=2, axis=None)

          self.derivativeVals = self.derivativeVals.reshape(1,-1) + np.zeros((self.number,self.samples))
          self.doubleDerivativeVals = self.doubleDerivativeVals.reshape(1,-1) + np.zeros((self.number,self.samples))
        
        if isinstance(self.surfaceVals,float):
            self.surfaceVals = self.surfaceVals + np.zeros((self.number,self.samples))
            self.derivativeVals = self.derivativeVals + np.zeros((self.number,self.samples))

        

        
        
        
    
    def surfaceChecker(self, relaxed = True):
        #self.doubleDerivativeVals = sp.misc.derivative(self.surfaceFunction,self.x,n=2)
        numerator = 1 + (self.derivativeVals)**2
        denominator = self.doubleDerivativeVals
        self.curvature = (numerator**1.5)/np.abs((denominator))
        self.condition = 1/((self.k*self.curvature)**0.333333333)
        print(self.condition.max())
        
        if self.condition.max() > 0.8660254037844386:
            print("Condition failed")
            print(self.condition.max())
        
            self.checker = False
        else:
            self.checker = True
        return self.checker
        
        
            
    def __CalculateAngle(self):
        ac = []
        ab = []
        for i in range(len(self.sourceLocationX[0])):
            ac.append([1,0])
            tempFrac = 1/np.sqrt(((self.x[0][i]-self.sourceLocationX[0][i])**2+
                           (self.surfaceVals[0][i]-self.sourceLocationY[0][i])**2))
            ab.append([tempFrac*(self.x[0][i]-self.sourceLocationX[0][i]),
                       tempFrac*(self.surfaceVals[0][i] - 
                                 self.sourceLocationY[0][i])])
        s = np.array(ac)
        r = np.array(ab)
        SdotR = np.einsum('ij,ij->i',s,r)
        modSmodR = np.linalg.norm(r,axis=1)*np.linalg.norm(s,axis=1)
        temp = np.abs(np.arccos(SdotR/modSmodR) - (-self.sourceAngle + np.pi/2))
        output = mb.repmat(temp,self.number,1)
        return output
        
        
            
    def __Integrand(self):
        
        r =  - self.x + self.sourceLocationX
        r2 =  - self.x + self.receiverLocationsX
        l = self.sourceLocationY - self.surfaceVals
        l2 = self.receiverLocationsY - self.surfaceVals
        R1 = np.sqrt( r*r + l*l )
        R2 = np.sqrt( r2*r2 + l2*l2 )
        qz = -self.k * ( l2*1/R2 + l*1/R1 )
        qx = -self.k *( (-self.derivativeVals*(-self.surfaceVals + self.sourceLocationY)+self.x-self.sourceLocationX)*1/R1
                     + (-self.derivativeVals*(-self.surfaceVals + self.receiverLocationsY)+self.x-self.receiverLocationsX)*1/R2)
        
        theta = np.arccos(l*1/R1) - (- self.sourceAngle + np.pi/2)
        #theta2 = self.__CalculateAngle()
        #Seems like theta2 is incorrect. This means I need to go back 
        #and change the others in the non-vectorised version
        Directivity = (sp.special.jn(1,self.k*self.a*np.sin(theta)))/(self.k*self.a*np.sin(theta))
        Directivity[np.isnan(Directivity)] = 0.5
        #Directivity2 = (sp.special.jn(1,self.k*self.a*np.sin(theta2)))/(self.k*self.a*np.sin(theta2))
        #Directivity2[np.isnan(Directivity2)] = 0.5 

        #G = (2*(qz)*Directivity*np.exp(1j*(R1+R2)*self.k)*1)/np.sqrt(R1*R2)
        que = qz - self.derivativeVals*qx
        F = 2*que*Directivity*np.exp(0+1j*(R1+R2)*self.k)*1/np.sqrt(R1*R2)
        
        return F
        
    def Scatter(self,absolute=False,norm=True):
        F  = self.__Integrand()
        if self.method == 'trapz':
            p = -1j/(2*np.pi*self.k)*np.trapz(F,self.x,axis=1)

            if norm == True:
                p = np.abs(p)/np.max(np.abs(p))
                return p
            elif absolute == True:
                p = np.abs(p)
                return p
            else:
                return p 
        elif self.method == 'simp':
            p = -1j/(2*np.pi*self.k)*sp.integrate.simps(F,self.x,axis=1)
            
            if norm == True:
                p = np.abs(p)/np.max(np.abs(p))
                return p
            elif absolute == True:
                p = np.abs(p)
                return p
            else:
                return p 
        
        elif self.method == 'cumtrapz':
            p = -1j/(2*np.pi*self.k)*sp.integrate.cumtrapz(F,self.x,axis=1)
            if norm == True:
                p = np.abs(p)/np.max(np.abs(p))
                return p
            elif absolute == True:
                p = np.abs(p)
                return p
            else:
                return p
            
def fourierRes(signal, timeIncrement,axis = 0):
    '''Just give us the fourier transform of a signal'''
    import scipy as sp
    sampleFrequency = 1/timeIncrement
    n = signal.shape[0]
    yf = sp.fft.fft(signal,axis=axis)
    #xf = sp.fft.fftshift(sp.fft.fftfreq(n,timeIncrement))
    xf = sp.fft.fftfreq(n,timeIncrement)
    return xf,yf

def spatialRes(signal):
    y = sp.fft.fft(signal,axis=0)
    return y

def Doppler(signals,dt,axis = 0):
    '''Takes signals of shape (number of signals, number of samples, number of receivers ).
    Returns the Doppler spectrum of all the receivers.'''
    n = np.array(signals[0]).size
    fSig = []
    
    for s in signals:
        x,Fsignals = fourierRes(np.array(s),dt,axis=axis)
        fSig.append(Fsignals)
    freq = x
    return freq,1/n*np.mean(np.abs(np.array(fSig))**2,axis=0)
