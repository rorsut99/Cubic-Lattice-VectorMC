# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 19:35:42 2023

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


Lx=6
Ly=6
Lz=6

#Coordination Number
coordNum=6
#Heisenberg Exchange
Jex=1.0

# #Magnetic field strength/direction +z-->phi=0, +x-->(phi=pi/2 and theta=0), +y-->(phi=pi/2,theta=pi/2)
# h=100.0
# Phi_mag=0
# Theta_mag=0




# #Anisotropic Coupling: rows of coupling strength matrix
# Axx=0.0
# Ayy=0.0
# Azz=0.0
# Axy=0.0
# Axz=0.0
# Ayz=0.0

# alpha=np.array([Axx,Axy,Axz])
# beta=np.array([Axy,Ayy,Ayz])
# gamma=np.array([Axz,Ayz,Azz])


N=Lx*Ly*Lz
Nsw=200000


T=0.1



@numba.njit(cache=True,fastmath=True)
def calcXPos(Phi,Theta):
    return(math.sin(Phi)*math.cos(Theta))

@numba.njit(cache=True,fastmath=True)
def calcYPos(Phi,Theta):
    return(math.sin(Phi)*math.sin(Theta))

@numba.njit(cache=True,fastmath=True)
def calcZPos(Phi,Theta):
    return(math.cos(Phi))

@numba.njit(cache=True,fastmath=True)
def dotProduct(x1,y1,z1,x2,y2,z2):
    return(x1*x2+y1*y2+z1*z2)


def initSpins():
    Xpos=np.zeros(((Lx,Ly,Lz)))
    Ypos=np.zeros(((Lx,Ly,Lz)))
    Zpos=np.zeros(((Lx,Ly,Lz)))

    for ix in range(0,Lx):
        for iy in range(0,Ly):
            for iz in range(0,Lx):
                Phi_0=np.arccos(np.random.uniform(-1.0,1.0))
                Theta_0=np.arccos(np.random.uniform(-1.0,1.0))
                quadTheta=random.choice([0,1])
                Theta_0=(2*np.pi-Theta_0 if(quadTheta==1) else Theta_0)
                
                Xpos[ix][iy][iz]=calcXPos(Phi_0,Theta_0)
                Ypos[ix][iy][iz]=calcYPos(Phi_0,Theta_0)
                Zpos[ix][iy][iz]=calcZPos(Phi_0,Theta_0)
                
    return(Xpos,Ypos,Zpos)


################################################################################################


@numba.njit(cache=True,fastmath=True)
def totalEnergy(Xpos,Ypos,Zpos):
    Etot=0.0
    for ix in range(0,Lx):
        for iy in range(0,Ly):
            for iz in range(0,Lz):
                ixP=(ix+1)%Lx
                iyP=(iy+1)%Ly
                izP=(iz+1)%Lz
                
                x1=Xpos[ix][iy][iz]
                y1=Ypos[ix][iy][iz]
                z1=Zpos[ix][iy][iz]
                
                
                #X plus spin
                x2=Xpos[ixP][iy][iz]
                y2=Ypos[ixP][iy][iz]
                z2=Zpos[ixP][iy][iz]
                
                
                #Y plus spin
                x3=Xpos[ix][iyP][iz]
                y3=Ypos[ix][iyP][iz]
                z3=Zpos[ix][iyP][iz]
                
                
                
                #Z plus spin
                x4=Xpos[ix][iy][izP]
                y4=Ypos[ix][iy][izP]
                z4=Zpos[ix][iy][izP]
                
                
                

                Ex=dotProduct(x1,y1,z1,x2,y2,z2)
                Ey=dotProduct(x1,y1,z1,x3,y3,z3)
                Ez=dotProduct(x1,y1,z1,x4,y4,z4)
                
                
                Etot+=Jex*(Ex+Ey+Ez)
    return(Etot)



@numba.njit(cache=True,fastmath=True)
def flipEnergy(Xpos,Ypos,Zpos,ix,iy,iz,xNew,yNew,zNew):
    ixP,ixM = (ix+1)%Lx, (ix-1)%Lx
    iyP,iyM = (iy+1)%Ly, (iy-1)%Ly
    izP,izM = (iz+1)%Lz, (iz-1)%Lz
    
    EoldX=dotProduct(Xpos[ix][iy][iz],Ypos[ix][iy][iz],Zpos[ix][iy][iz],Xpos[ixP][iy][iz],Ypos[ixP][iy][iz],Zpos[ixP][iy][iz])
    EoldX+=dotProduct(Xpos[ix][iy][iz],Ypos[ix][iy][iz],Zpos[ix][iy][iz],Xpos[ixM][iy][iz],Ypos[ixM][iy][iz],Zpos[ixM][iy][iz])
    
    EoldY=dotProduct(Xpos[ix][iy][iz],Ypos[ix][iy][iz],Zpos[ix][iy][iz],Xpos[ix][iyP][iz],Ypos[ix][iyP][iz],Zpos[ix][iyP][iz])
    EoldY+=dotProduct(Xpos[ix][iy][iz],Ypos[ix][iy][iz],Zpos[ix][iy][iz],Xpos[ix][iyM][iz],Ypos[ix][iyM][iz],Zpos[ix][iyM][iz])
    
    EoldZ=dotProduct(Xpos[ix][iy][iz],Ypos[ix][iy][iz],Zpos[ix][iy][iz],Xpos[ix][iy][izP],Ypos[ix][iy][izP],Zpos[ix][iy][izP])
    EoldZ+=dotProduct(Xpos[ix][iy][iz],Ypos[ix][iy][iz],Zpos[ix][iy][iz],Xpos[ix][iy][izM],Ypos[ix][iy][izM],Zpos[ix][iy][izM])
    
    Eold=EoldX+EoldY+EoldZ
    
    
    EnewX=dotProduct(xNew,yNew,zNew,Xpos[ixP][iy][iz],Ypos[ixP][iy][iz],Zpos[ixP][iy][iz])
    EnewX+=dotProduct(xNew,yNew,zNew,Xpos[ixM][iy][iz],Ypos[ixM][iy][iz],Zpos[ixM][iy][iz])
    
    EnewY=dotProduct(xNew,yNew,zNew,Xpos[ix][iyP][iz],Ypos[ix][iyP][iz],Zpos[ix][iyP][iz])
    EnewY+=dotProduct(xNew,yNew,zNew,Xpos[ix][iyM][iz],Ypos[ix][iyM][iz],Zpos[ix][iyM][iz])
    
    EnewZ=dotProduct(xNew,yNew,zNew,Xpos[ix][iy][izP],Ypos[ix][iy][izP],Zpos[ix][iy][izP])
    EnewZ+=dotProduct(xNew,yNew,zNew,Xpos[ix][iy][izM],Ypos[ix][iy][izM],Zpos[ix][iy][izM])
    
    Enew=EnewX+EnewY+EnewZ
    
    return(Jex*(Enew-Eold))
    
###################################################################################################################

# @numba.njit
# def totEnergyMag(h,Xpos,Ypos,Zpos,xPosMag,yPosMag,zPosMag):
#     Emag=0.0
#     for ix in range(0,Lx):
#         for iy in range(0,Ly):
#             for iz in range(0,Lz):
#                 Emag+=h*dotProduct(Xpos[ix][iy][iz],Ypos[ix][iy][iz],Zpos[ix][iy][iz],xPosMag,yPosMag,zPosMag)
#     return(Emag)

# @numba.njit              
# def flipEnergyMag(xOld,yOld,zOld,xNew,yNew,zNew,xPosMag,yPosMag,zPosMag):
#     Eold=dotProduct(xOld,yOld,zOld,xPosMag,yPosMag,zPosMag)
#     Enew=dotProduct(xNew,yNew,zNew,xPosMag,yPosMag,zPosMag)
#     return(h*(Enew-Eold))

# ##################################################################################################################

# @numba.njit
# def aniEnergy(alpha,beta,gamma,X1,Y1,Z1):
#     EAni=0.0

    
#     EAni+=alpha[0]*X1*X1+alpha[1]*X1*Y1+alpha[2]*X1*Z1
#     EAni+=beta[0]*Y1*X1+beta[1]*Y1*Y1+beta[2]*Y1*Z1
#     EAni+=gamma[0]*Z1*X1+gamma[1]*Z1*Y1+gamma[2]*Z1*Z1
#     return(EAni)

# @numba.njit
# def flipEnergyAni(alpha,beta,gamma,xOld,yOld,zOld,xNew,yNew,zNew):
#     Eold=aniEnergy(alpha,beta,gamma,xOld,yOld,zOld)
#     Enew=aniEnergy(alpha,beta,gamma,xNew,yNew,zNew)
#     return(Enew-Eold)


# @numba.njit
# def totEnergyAni(alpha,beta,gamma,xPos,yPos,zPos):
#     EAniTot=0.0
#     for ix in range(0,Lx):
#         for iy in range(0,Ly):
#             for iz in range(0,Lz):
#                 EAniTot+=aniEnergy(alpha,beta,gamma,xPos[ix][iy][iz],yPos[ix][iy][iz],zPos[ix][iy][iz])
#     return(EAniTot)


# #############################################################################################################

# @numba.njit 
# def calcMag(xPos,yPos,zPos,xMagDir,yMagDir,zMagDir):
#     mag=0.0
#     for ix in range(0,Lx):
#         for iy in range(0,Ly):
#             for iz in range(0,Lz):
#                 mag+=dotProduct(xPos[ix][iy][iz],yPos[ix][iy][iz],zPos[ix][iy][iz],xMagDir,yMagDir,zMagDir)
#     return(mag)

################################################################################################################
@numba.njit(cache=True,fastmath=True)
def meanField(xPos,yPos,zPos,ix,iy,iz):
    ixP,ixM = (ix+1)%Lx, (ix-1)%Lx
    iyP,iyM = (iy+1)%Ly, (iy-1)%Ly
    izP,izM = (iz+1)%Lz, (iz-1)%Lz
    meanX=xPos[ixP][iy][iz]+xPos[ixM][iy][iz]+xPos[ix][iyP][iz]+xPos[ix][iyM][iz]+xPos[ix][iy][izP]+xPos[ix][iy][izM]
    meanY=yPos[ixP][iy][iz]+yPos[ixM][iy][iz]+yPos[ix][iyP][iz]+yPos[ix][iyM][iz]+yPos[ix][iy][izP]+yPos[ix][iy][izM]
    meanZ=zPos[ixP][iy][iz]+zPos[ixM][iy][iz]+zPos[ix][iyP][iz]+zPos[ix][iyM][iz]+zPos[ix][iy][izP]+zPos[ix][iy][izM]
    return(-meanX/coordNum,-meanY/coordNum,-meanZ/coordNum)

@numba.njit(cache=True,fastmath=True)
def deterministicUpdate(xPos,yPos,zPos):
    for t in range(0,N):
        ix, iy, iz=random.randint(0,Lx-1), random.randint(0,Ly-1),random.randint(0,Lz-1)

        xPos[ix][iy][iz],yPos[ix][iy][iz],zPos[ix][iy][iz]=meanField(xPos,yPos,zPos,ix,iy,iz)
    return(xPos,yPos,zPos)
    


############################################################################################################3####

@numba.njit(cache=True,fastmath=True)
def simulate(xPos,yPos,zPos,T):
    Etot=totalEnergy(xPos,yPos,zPos)
    # Etot-=totEnergyMag(h,xPos,yPos,zPos,xPosMag,yPosMag,zPosMag)
    # Etot-=totEnergyAni(alpha,beta,gamma,xPos,yPos,zPos)

    B=1.0/T
    for t in range(0,Nsw):
        for t0 in range(0,N):
            ix, iy, iz=random.randint(0,Lx-1), random.randint(0,Ly-1),random.randint(0,Lz-1)
            Phi_prop=math.acos(random.uniform(-1.0,1.0))
            
            Theta_0=math.acos(random.uniform(-1.0,1.0))
            quadTheta=random.uniform(0,1)
            Theta_prop=(2*np.pi-Theta_0 if(quadTheta<0.5) else Theta_0)
            
            xProp=calcXPos(Phi_prop,Theta_prop)
            yProp=calcYPos(Phi_prop,Theta_prop)
            zProp=calcZPos(Phi_prop,Theta_prop)
            
            
            dE=flipEnergy(xPos,yPos,zPos,ix,iy,iz,xProp,yProp,zProp)
            # dE-=flipEnergyMag(xPos[ix][iy][iz],yPos[ix][iy][iz],zPos[ix][iy][iz],xProp,yProp,zProp,xPosMag,yPosMag,zPosMag)
            # dE-=flipEnergyAni(alpha,beta,gamma,xPos[ix][iy][iz],yPos[ix][iy][iz],zPos[ix][iy][iz],xProp,yProp,zProp)
            
            check=(dE < 0 or  B*dE < -np.log(random.uniform(0,1)))
            if check:
                
                xPos[ix][iy][iz]=xProp
                yPos[ix][iy][iz]=yProp
                zPos[ix][iy][iz]=zProp
                Etot+=dE
                           
    return(Etot,xPos,yPos,zPos)



################################################################################################################3



# xPosMag=calcXPos(Phi_mag,Theta_mag)
# yPosMag=calcYPos(Phi_mag,Theta_mag)
# zPosMag=calcZPos(Phi_mag,Theta_mag)


xPos,yPos,zPos=initSpins()
# E1=totalEnergy(xPos,yPos,zPos)

Esim,xPos,yPos,zPos=simulate(xPos,yPos,zPos,T)
# E2=totalEnergy(xPos,yPos,zPos)
print(totalEnergy(xPos,yPos,zPos)/N)
xPos,yPos,zPos=deterministicUpdate(xPos, yPos, zPos)

# checkMagX=0.0
# checkMagY=0.0
# checkMagZ=1.0

# mag=calcMag(xPos,yPos,zPos,checkMagX,checkMagY,checkMagZ)

print(Esim/N)
print(totalEnergy(xPos,yPos,zPos)/N)

end=time.time()
print('Runtime:',end-start)







