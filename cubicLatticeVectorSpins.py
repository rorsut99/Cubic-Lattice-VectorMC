# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:17:41 2023

@author: rorst
"""

import numpy as np 
import random
import numba 
import math
import sys
import json
import time

seed = random.randrange(sys.maxsize)
random.seed(seed)

start=time.time()

# Lx=int(sys.argv[1])
# Ly=int(sys.argv[2])
# Lz=int(sys.argv[3])
# Nsw=int(sys.argv[4])


Lx=4
Ly=4
Lz=4

#Heisenberg Exchange
Jex=1.0

#Magnetic field strength/direction +z-->phi=0, +x-->(phi=pi/2 and theta=0), +y-->(phi=pi/2,theta=pi/2)
h=0.0
Phi_mag=0
Theta_mag=0


#Anisotropic Coupling: rows of coupling strength matrix
Axx=0.0
Ayy=0.0
Azz=0.0
Axy=0.0
Axz=0.0
Ayz=0.0

alpha=np.array([Axx,Axy,Axz])
beta=np.array([Axy,Ayy,Ayz])
gamma=np.array([Axz,Ayz,Azz])


N=Lx*Ly*Lz
Nsw=10000


T=0.1

#############################################################################################3

@numba.njit
def dotProduct(theta1,theta2,phi1,phi2):
    return(((math.cos(theta1-theta2)*math.sin(phi1)*math.sin(phi2))+(math.cos(phi1)*math.cos(phi2))))

def initSpins():
    phi=np.zeros(((Lx,Ly,Lz)))
    theta=np.zeros(((Lx,Ly,Lz)))
    for ix in range(0,Lx):
        for iy in range(0,Ly):
            for iz in range(0,Lx):
                Phi_0=np.arccos(np.random.uniform(-1.0,1.0))
                phi[ix][iy][iz]=(Phi_0)
                
                
                Theta_0=np.arccos(np.random.uniform(-1.0,1.0))
                quadTheta=random.choice([0,1])
                theta[ix][iy][iz]=(2*np.pi-Theta_0 if(quadTheta==1) else Theta_0)
    return(phi,theta)


########################################################################################


@numba.njit
def totEnergyMag(h,thetaM,phiM,theta,phi):
    Emag=0.0
    for ix in range(0,Lx):
        for iy in range(0,Ly):
            for iz in range(0,Lz):
                Emag+=h*dotProduct(thetaM,theta[ix][iy][iz],phiM,phi[ix][iy][iz])
    return(Emag)

@numba.njit              
def flipEnergyMag(thetaOld,phiOld,thetaNew,phiNew,thetaM,phiM):
    Eold=dotProduct(thetaOld,thetaM,phiOld,phiM)
    Enew=dotProduct(thetaNew,thetaM,phiNew,phiM)
    return(h*(Enew-Eold))

##########################################################################################

@numba.njit
def aniEnergy(alpha,beta,gamma,theta1,phi1):
    EAni=0.0
    Xcomp=dotProduct(theta1,0.0,phi1,np.pi/2)
    Ycomp=dotProduct(theta1,np.pi/2,phi1,np.pi/2)
    Zcomp=dotProduct(theta1,0.0,phi1,0.0)
    
    
    EAni+=alpha[0]*Xcomp*Xcomp+alpha[1]*Xcomp*Ycomp+alpha[2]*Xcomp*Zcomp
    EAni+=beta[0]*Ycomp*Xcomp+beta[1]*Ycomp*Ycomp+beta[2]*Ycomp*Zcomp
    EAni+=gamma[0]*Zcomp*Xcomp+gamma[1]*Zcomp*Ycomp+gamma[2]*Zcomp*Zcomp
    return(EAni)

@numba.njit
def flipEnergyAni(alpha,beta,gamma,thetaOld,phiOld,thetaNew,phiNew):
    Eold=aniEnergy(alpha,beta,gamma,thetaOld,phiOld)
    Enew=aniEnergy(alpha,beta,gamma,thetaNew,phiNew)
    return(Enew-Eold)


@numba.njit
def totEnergyAni(alpha,beta,gamma,theta,phi):
    EAniTot=0.0
    for ix in range(0,Lx):
        for iy in range(0,Ly):
            for iz in range(0,Lz):
                EAniTot+=aniEnergy(alpha,beta,gamma,theta[ix][iy][iz],phi[ix][iy][iz])
    return(EAniTot)


#################################################################################################


@numba.njit 
def calcMag(theta,phi,thetaDir,phiDir):
    mag=0.0
    for ix in range(0,Lx):
        for iy in range(0,Ly):
            for iz in range(0,Lz):
                mag+=dotProduct(theta[ix][iy][iz],thetaDir,phi[ix][iy][iz],phiDir)
    return(mag)


##############################################################################################


@numba.njit
def totalEnergy(theta,phi):
    Etot=0.0
    for ix in range(0,Lx):
        for iy in range(0,Ly):
            for iz in range(0,Lz):
                ixP=(ix+1)%Lx
                iyP=(iy+1)%Ly
                izP=(iz+1)%Lz
                
                theta1=theta[ix][iy][iz]
                phi1=phi[ix][iy][iz]
                
                
                thetaX=theta[ixP][iy][iz]
                phiX=phi[ixP][iy][iz]
                Ex=dotProduct(theta1,thetaX,phi1,phiX)
                
                
                thetaY=theta[ix][iyP][iz]
                phiY=phi[ix][iyP][iz]
                Ey=dotProduct(theta1,thetaY,phi1,phiY)
                
                thetaZ=theta[ix][iy][izP]
                phiZ=phi[ix][iy][izP]
                Ez=dotProduct(theta1,thetaZ,phi1,phiZ)
                
                Etot+=Jex*(Ex+Ey+Ez)
    return(Etot)



@numba.njit
def flipEnergy(theta,phi,ix,iy,iz,thetaNew,phiNew):
    ixP,ixM = (ix+1)%Lx, (ix-1)%Lx
    iyP,iyM = (iy+1)%Ly, (iy-1)%Ly
    izP,izM = (iz+1)%Lz, (iz-1)%Lz
    
    EoldX=dotProduct(theta[ix][iy][iz],theta[ixP][iy][iz],phi[ix][iy][iz],phi[ixP][iy][iz])+ dotProduct(theta[ix][iy][iz],theta[ixM][iy][iz],phi[ix][iy][iz],phi[ixM][iy][iz])
    
    EoldY=dotProduct(theta[ix][iy][iz],theta[ix][iyP][iz],phi[ix][iy][iz],phi[ix][iyP][iz])+ dotProduct(theta[ix][iy][iz],theta[ix][iyM][iz],phi[ix][iy][iz],phi[ix][iyM][iz])
    
    EoldZ=dotProduct(theta[ix][iy][iz],theta[ix][iy][izP],phi[ix][iy][iz],phi[ix][iy][izP])+ dotProduct(theta[ix][iy][iz],theta[ix][iy][izM],phi[ix][iy][iz],phi[ix][iy][izM])
    
    Eold=EoldX+EoldY+EoldZ
    
    
    EnewX=dotProduct(thetaNew,theta[ixP][iy][iz],phiNew,phi[ixP][iy][iz])+ dotProduct(thetaNew,theta[ixM][iy][iz],phiNew,phi[ixM][iy][iz])
    
    EnewY=dotProduct(thetaNew,theta[ix][iyP][iz],phiNew,phi[ix][iyP][iz])+ dotProduct(thetaNew,theta[ix][iyM][iz],phiNew,phi[ix][iyM][iz])
    
    EnewZ=dotProduct(thetaNew,theta[ix][iy][izP],phiNew,phi[ix][iy][izP])+ dotProduct(thetaNew,theta[ix][iy][izM],phiNew,phi[ix][iy][izM])
    
    Enew=EnewX+EnewY+EnewZ
    
    return(Jex*(Enew-Eold))
    

###########################################################################################################################


@numba.njit
def simulate(theta,phi,T):
    Etot=totalEnergy(theta,phi)
    Etot-=totEnergyMag(h,Theta_mag,Phi_mag,theta,phi)
    Etot-=totEnergyAni(alpha,beta,gamma,theta,phi)

    B=1.0/T
    for t in range(0,Nsw):
        for t0 in range(0,N):
            ix, iy, iz=random.randint(0,Lx-1), random.randint(0,Ly-1),random.randint(0,Lz-1)
            Phi_prop=np.arccos(np.random.uniform(-1.0,1.0))
            
            Theta_0=np.arccos(np.random.uniform(-1.0,1.0))
            quadTheta=random.uniform(0,1)
            Theta_prop=(2*np.pi-Theta_0 if(quadTheta<0.5) else Theta_0)
            
            dE=flipEnergy(theta,phi,ix,iy,iz,Theta_prop,Phi_prop)
            dE-=flipEnergyMag(theta[ix][iy][iz],phi[ix][iy][iz],Theta_prop,Phi_prop,Theta_mag,Phi_mag)
            dE-=flipEnergyAni(alpha,beta,gamma,theta[ix][iy][iz],phi[ix][iy][iz],Theta_prop,Phi_prop)
            
            check=(dE < 0 or  B*dE < -np.log(random.uniform(0,1)))
            if check:

                theta[ix][iy][iz]=Theta_prop
                phi[ix][iy][iz]=Phi_prop    
                Etot+=dE
                           
    return(Etot,theta,phi)
        
    
                

################################################################################################



phi,theta=initSpins()
E1=totalEnergy(theta,phi)

Esim,theta,phi=simulate(theta,phi,T)
E2=totalEnergy(theta,phi)

mag=calcMag(theta, phi, 0, 0)

print(Esim/N)
# print(mag/N)
end=time.time()
print('Runtime:',end-start)
































