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
import corner as corner
import csv
import time

class Directed3DVectorised:
    def __init__(self,sourceLocation,receiverLocations,surfaceFunction,frequency,a,sourceAngle = -np.pi/4,method = 'trapz',userMinMax = None,userSamples = 9000,absolute = True):
        
        '''
        Init function for the Kirchhoff approximation using a radiation from a baffled piston source.
        Inputs:
        
        sourceLocation: List of x and y component of the source location
        receiverLocations: List of lists of x and y components of receiver locations
        surfaceFunction: Either a function that only takes x as an argument, or an array of values that relate to surface elevation.
        frequency: Source frequency 
        a: Piston aperture
        sourceAngle: Angle of the source
        method: Integration methods, can be trapz, simps or cumtrapz. simp is usually recommended
        userMinMax: [a,b] start and end point of the surface
        userSamples: integer showing density of samples
        absolute: Bool, absolute value
        '''
        
        self.sourceLocationX = np.array(sourceLocation)[0]
        self.sourceLocationY = np.array(sourceLocation)[1]
        self.sourceLocationZ = np.array(sourceLocation)[2]
        self.receiverLocationsX = np.array(receiverLocations)[:,0]
        self.receiverLocationsY = np.array(receiverLocations)[:,1]
        self.receiverLocationsZ = np.array(receiverLocations)[:,2]
        self.k = (2*np.pi*frequency)/343
        self.a = a
        
        theta = sourceAngle[0]
        phi = sourceAngle[1]
        
        base_pointing = np.array([1,0,0])
        base_pointing = (base_pointing) / np.linalg.norm(base_pointing)
        
        angle_pointing_vec = np.array([np.cos(phi)*np.sin(theta),np.sin(phi)*np.sin(theta), np.cos(theta)])
        angle_pointing_vec = angle_pointing_vec / np.linalg.norm(angle_pointing_vec)
        
        angle = np.math.acos(np.dot(base_pointing,angle_pointing_vec))
        
        self.sourceAngle = angle
        self.surfaceFunction = surfaceFunction
        self.method = method
        self.absolute = absolute
        if userMinMax == None:
            self.min = self.receiverLocationsX.min()
            self.max = self.receiverLocationsX.max()
        else:
            self.minX = userMinMax[0][0]
            self.maxX = userMinMax[0][1]
            
            self.minY = userMinMax[1][0]
            self.maxY = userMinMax[1][1]
            
        self.number = self.receiverLocationsX.shape[0]
        self.samples = userSamples

        self.receiverLocationsX = self.receiverLocationsX[:,np.newaxis,np.newaxis]
        self.receiverLocationsY = self.receiverLocationsY[:,np.newaxis,np.newaxis]
        self.receiverLocationsZ = self.receiverLocationsZ[:,np.newaxis,np.newaxis]
        
        self.sourceLocationX = self.sourceLocationX + np.zeros((self.number, self.samples[0], self.samples[1]))
        self.sourceLocationY = self.sourceLocationY + np.zeros((self.number, self.samples[0], self.samples[1]))
        self.sourceLocationZ = self.sourceLocationZ + np.zeros((self.number, self.samples[0], self.samples[1]))

        self.receiverLocationsX = np.tile(self.receiverLocationsX,(1,self.samples[0], self.samples[1]))
        self.receiverLocationsY = np.tile(self.receiverLocationsY,(1,self.samples[0], self.samples[1]))
        self.receiverLocationsZ = np.tile(self.receiverLocationsZ,(1,self.samples[0], self.samples[1]))

        
        if type(self.surfaceFunction) == np.ndarray:
            pass
            X = np.linspace(self.minX,self.maxY,self.samples[0])
            Y = np.linspace(self.minY,self.maxY,self.samples[1])
            self.grid = np.meshgrid(X,Y)
            
            self.gridY = np.tile(self.grid[1],(self.number,1,1))
            self.gridX = np.tile(self.grid[0],(self.number,1,1))
            
            self.surfaceVals = self.surfaceFunction
            
            self.derivativeValsX = np.gradient(self.surfaceVals,self.grid[0][0][1] - self.grid[0][0][0],edge_order=2, axis=0)
            self.derivativeValsY = np.gradient(self.surfaceVals,self.grid[1][1][0] - self.grid[1][0][0],edge_order=2, axis=1)
            
            self.doubleDerivativeValsX = np.gradient(self.derivativeValsX,self.grid[0][0][1] - self.grid[0][0][0],edge_order=2, axis=0)
            self.doubleDerivativeValsY = np.gradient(self.derivativeValsY,self.grid[1][1][0] - self.grid[1][0][0],edge_order=2, axis=1)
        
            

        else:
            
            #Grid is of form (2, self.samples, self.samples), the 2 corresponds to the x value and the y values.
            X = np.linspace(self.minX,self.maxY,self.samples[0])
            Y = np.linspace(self.minY,self.maxY,self.samples[1])
            self.grid = np.meshgrid(X,Y)
            
            self.gridY = np.tile(self.grid[1],(self.number,1,1))
            self.gridX = np.tile(self.grid[0],(self.number,1,1))
            
            self.surfaceVals = surfaceFunction(self.grid)
    
            #These look like they may be the wrong way round
            self.derivativeValsX = np.gradient(self.surfaceVals,self.grid[0][0][1] - self.grid[0][0][0],edge_order=2, axis=0)
            self.derivativeValsY = np.gradient(self.surfaceVals,self.grid[1][1][0] - self.grid[1][0][0],edge_order=2, axis=1)
            
            self.doubleDerivativeValsX = np.gradient(self.derivativeValsX,self.grid[0][0][1] - self.grid[0][0][0],edge_order=2, axis=0)
            self.doubleDerivativeValsY = np.gradient(self.derivativeValsY,self.grid[1][1][0] - self.grid[1][0][0],edge_order=2, axis=1)
    
    def __Integrand(self):


        r =  - self.gridX + self.sourceLocationX 
        r2 =  - self.gridX + self.receiverLocationsX
        m = self.sourceLocationZ - self.surfaceVals
        m2 = self.receiverLocationsZ - self.surfaceVals
        l = - self.gridY + self.sourceLocationY
        l2 = - self.gridY + self.receiverLocationsY
        R1 = np.sqrt( r*r + l*l + m*m )
        R2 = np.sqrt( r2*r2 + l2*l2 + m2*m2 )
        
        #From Maple
        qz = -self.k * ( -m*1/R1 - m2*1/R2)   
        qx = -self.k * (( self.derivativeValsX * (-m) - r) / R1 +
                        ( self.derivativeValsX * (-m2) - r2) / R2)
        
        qy = -self.k * (( self.derivativeValsY * (-m) - l) / R1 +
                        ( self.derivativeValsY * (-m2) - l2) / R2)
        
        
        #This could be wrong

        theta = np.arccos(m*1/R1) - (- self.sourceAngle)# + np.pi/2)

        Directivity = (sp.special.jn(1,self.k*self.a*np.sin(theta)))/(self.k*self.a*np.sin(theta))
        Directivity[np.isnan(Directivity)] = 0.5

        que = qz - self.derivativeValsX*qx - self.derivativeValsY*qy
        F = 2*que*Directivity*np.exp(0+1j*(R1+R2)*self.k)*1/(R1*R2)
        
        return F
        
    
    
    
    
    def Scatter(self,absolute=False,norm=True,direct_field=False):
        F  = self.__Integrand()
        p = np.zeros(F.shape[0],dtype=np.complex128)
        
        

        if direct_field:
            r =  - self.receiverLocationsX.T[0] + self.sourceLocationX.T[0]
            l = - self.receiverLocationsY.T[0] + self.sourceLocationY.T[0]
            m = - self.receiverLocationsZ.T[0] + self.sourceLocationZ.T[0]
            R1 = np.sqrt( r*r + l*l + m*m ) 
            theta = np.arccos(m*1/R1) - (- self.sourceAngle + np.pi/2)
            Directivity = (sp.special.jn(1,self.k*self.a*np.sin(theta)))/(self.k*self.a*np.sin(theta))
            Directivity[np.isnan(Directivity)] = 0.5
            #This may be needed for boundary conditions
            #Directivity[ 3/2*np.pi > theta > np.pi/2] = 0

            field = 2*Directivity*np.exp(0+1j*(R1)*self.k)*1/(R1)
            p += field
        if self.method == 'trapz':
            
            interior = np.trapz(F,self.gridY[0].T[0],axis=2)
            p += -1j/(4*np.pi)*np.trapz(interior,self.gridX[0][0],axis=1)

            if norm == True:
                p = np.abs(p)/np.max(np.abs(p))
                return p
            elif absolute == True:
                p = np.abs(p)
                return p
            else:
                return p 
        elif self.method == 'simp':

            interior = sp.integrate.simps(F,self.gridY[0].T[0],axis=2)
            p += -1j/(4*np.pi)*sp.integrate.simps(interior,self.gridX[0][0],axis=1)
            
            if norm == True:
                p = np.abs(p)/np.max(np.abs(p))
                return p

            
            elif absolute == True:
                p = np.abs(p)
                return p
            else:
                return p 

            
        elif self.method == 'cumtrapz':
            interior = sp.integrate.cumtrapz(F,self.gridY[0].T[0],axis=2)
            p += -1j/(4*np.pi)*sp.integrate.cumtrapz(interior,self.gridX[0][0],axis=1)
            if norm == True:
                p = np.abs(p)/np.max(np.abs(p))
                return p
            elif absolute == True:
                p = np.abs(p)
                return p
            else:
                return p
