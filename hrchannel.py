import numpy as np
import math

def hr_channel(N1,N2,K,Lc,L):
    N=N1*N2
    d=0.5
    hK=np.zeros((N,K),dtype=complex)

    N1index=np.arange(-(N1-1)/2,(N1/2),1)
    N1index=N1index.reshape(len(N1index),1)*(2/N1)
    N2index=np.arange(-(N2-1)/2,(N2/2),1)
    N2index=N2index.reshape(len(N2index),1)*(2/N2)

    index=np.random.permutation(N)
    x=np.ceil(index[0:Lc]/N2)
    x=x.astype(int)
    y=index[0:Lc]-(N2*(x-1))
    for i in range(len(x)):
        if(x[i]==16):
            x[i]=15
    phi1c=[]
    for i in x:
        phi1c.append(N1index[i])

    for i in range(len(y)):
        if(y[i]==16):
            y[i]=15
    phi2c=np.array([])
    for j in y:
        phi2c=np.append(phi2c,N2index[j])

    for k in range(K):
        alpha=np.zeros((L,1),dtype=complex)
        alpha[0:L] = np.random.normal(loc=0,scale=1,size=(L,1)) + 1j*np.random.normal(loc=0,scale=1,size=(L,1)) / np.sqrt(2)
        hr=np.zeros((N,1),dtype=int)

        phi1=phi1c[0:Lc]               
        phi2=phi2c[0:Lc]
        index=np.random.permutation(N)
        x=np.ceil(index[0:L-Lc]/N2)
        x=x.astype(int)
        y=index[0:L-Lc]-(N2*(x-1))
        for i,j in zip(range(Lc,L),x):
            phi1[i]=N1index[j]

        for k,m in zip(range(Lc,L),y):
            phi2[k]=N2index[y]

        for l in range(L):
            a1=(1/math.sqrt(N1))*np.exp((-1j)*2*np.pi*(np.arange(0,N1,1).reshape(N1,1))*d*phi1[l])
            a2=(1/math.sqrt(N2))*np.exp((-1j)*2*np.pi*(np.arange(0,N2,1).reshape(N2,1))*d*phi2[l])
            a=np.kron(a1,a2)
            hr= hr + alpha[l]*a
        
        hK[:,[k]]=math.sqrt(N/L)*hr
        
    return hK



