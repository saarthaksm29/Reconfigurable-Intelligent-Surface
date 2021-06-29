import numpy as np
import math
from matplotlib import pyplot as plt
from gchannel import *
from hrchannel import *
from dsomp import *

M1=8
M2=8
N1=16
N2=16
M=M1*M2 
N=N1*N2 
K=16    

L1=5    
L2=8    

L2c1=0  
L2c2=4  
L2c3=6  
L2c4=8

Q_all=np.arange(32,144,16)
d=0.5

C0 = 0.001
d1 = 10
d2 = 100
alpha1 = -2.2
alpha2 = -2.8
p1=C0*pow(d1,alpha1)
p2=C0*pow(d2,alpha2)

UN1=(1/math.sqrt(N1))*np.exp((-1j)*2*np.pi*(np.arange(0,N1,1).reshape(N1,1))*d*np.arange(-(N1-1)/2,(N1/2),1)*(2/N1))
UN2=(1/math.sqrt(N2))*np.exp((-1j)*2*np.pi*(np.arange(0,N2,1).reshape(N2,1))*d*np.arange(-(N2-1)/2,(N2/2),1)*(2/N2))
UN=np.kron(UN1,UN2)

UM1=(1/math.sqrt(M1))*np.exp((-1j)*2*np.pi*(np.arange(0,M1,1).reshape(M1,1))*d*np.arange(-(M1-1)/2,(M1/2),1)*(2/M1))
UM2=(1/math.sqrt(M2))*np.exp((-1j)*2*np.pi*(np.arange(0,M2,1).reshape(M2,1))*d*np.arange(-(M2-1)/2,(M2/2),1)*(2/M2))
UM=np.kron(UM1,UM2)

SNR_dB=0
SNR_linear=10**(SNR_dB/10)

sample=200
length=len(Q_all)

error0=np.zeros((sample,length),dtype=complex)
error1=np.zeros((sample,length),dtype=complex)
error2=np.zeros((sample,length),dtype=complex)
error3=np.zeros((sample,length),dtype=complex)
error4=np.zeros((sample,length),dtype=complex)
error5=np.zeros((sample,length),dtype=complex)
error6=np.zeros((sample,length),dtype=complex)
energy=np.zeros((sample,1),dtype=complex)

for s in range(1,sample):
    sigma2=(p1*p2)/(SNR_linear)
    G_new=np.array([])
    G_new=G_channel(M1,M2,N1,N2,L1)
    print(G_new.shape)
    G=[i*math.sqrt(p1) for i in G_new]
    # G=math.sqrt(p1)*G_channel(M1,M2,N1,N2,L1)
    hK=math.sqrt(p2)*hr_channel(N1,N2,K,L2,L2)
    H=np.zeros((N,M,K),dtype=float)
    for k in range(K):
        hr=hK[:,k]
        HC=G*np.diag(hr)
        H[k,:,:]=np.matmul(UM.T,HC,UN.T).T

    for iQ in range(length):
        Q=Q_all(iQ)
        Y=np.zeros((Q,M,K),dtype=int)
        W=((np.random(N,Q)>0.5)*2-1)/np.sqrt(N)
        A=np.matmul(UN*W).T
        
        for k in range(K):
            noise = np.sqrt(sigma2)*(np.random.normal(Q,M)+1j*np.random.normal(Q,M))/math.sqrt(2)
            Y[k,:,:]=A*H[k,:,:]+noise


        [Hhat3,row3,column3]=DS_OMP(Y,A,M,N,K,L1,L2,L2c1)

        [Hhat4,row4,column4]=DS_OMP(Y,A,M,N,K,L1,L2,L2c2)

        [Hhat5,row5,column5]=DS_OMP(Y,A,M,N,K,L1,L2,L2c3)

        [Hhat6,row6,column6]=DS_OMP(Y,A,M,N,K,L1,L2,L2c4)

        
        error3[s,iQ]=np.sum(np.sum(np.sum(np.square(np.absolute(Hhat3-H)),axis=0),axis=0),axis=0)
        error4[s,iQ]=np.sum(np.sum(np.sum(np.square(np.absolute(Hhat4-H)),axis=0),axis=0),axis=0)
        error5[s,iQ]=np.sum(np.sum(np.sum(np.square(np.absolute(Hhat5-H)),axis=0),axis=0),axis=0)
        error6[s,iQ]=np.sum(np.sum(np.sum(np.square(np.absolute(Hhat6-H)),axis=0),axis=0),axis=0)

    energy[s]=np.sum(np.sum(np.sum(np.square(np.absolute(H)),axis=0),axis=0),axis=0)

nmse3=np.mean(error3)/np.mean(energy)
nmse4=np.mean(error4)/np.mean(energy)
nmse5=np.mean(error5)/np.mean(energy)
nmse6=np.mean(error6)/np.mean(energy)

nmse3=10*math.log10(nmse3)
nmse4=10*math.log10(nmse4)
nmse5=10*math.log10(nmse5)
nmse6=10*math.log10(nmse6)

plt.grid(True)
plt.plot(Q_all,nmse3,'r<-',linewidth=1.5)
plt.plot(Q_all,nmse4,'ro-',linewidth=1.5)
plt.plot(Q_all,nmse5,'rs-',linewidth=1.5)
plt.plot(Q_all,nmse6,'rd-',linewidth=1.5)

plt.xlabel('The pilot overhead Q for the cascaded channel estimation')
plt.ylabel('NMSE (dB)')
plt.axis([32,128,-25,-5])
plt.tight_layout()

plt.show()