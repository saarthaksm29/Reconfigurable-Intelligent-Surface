import numpy as np
import math

def G_channel(M1,M2,N1,N2,L):
    M=M1*M2
    N=N1*N2

    d=0.5
    G=np.zeros((M,N),dtype=complex)

    M1index=np.arange(-(N1-1)/2,(M1/2),1)
    M2index=np.arange(-(N2-1)/2,(M2/2),1)
    N1index=np.arange(-(N1-1)/2,(N1/2),1)
    N2index=np.arange(-(N2-1)/2,(N2/2),1)

    M1index=M1index.reshape(1,len(M1index))
    M2index=M2index.reshape(1,len(M2index))
    N1index=N1index.reshape(1,len(N1index))
    N2index=N2index.reshape(1,len(N2index))

    M1index=M1index.T*(2/M1)
    M2index=M2index.T*(2/M2)
    N1index=N1index.T*(2/N1)
    N2index=N2index.T*(2/N2)

    index=np.random.permutation(N)
    x=np.ceil(index[0:L]/N2)
    x=x.astype(int)
    y=index[0:L]-(N2*(x-1))
    phi1=[]
    for k in range(len(x)):
        if(x[k]==16):
            x[k]=15
    for i in x:
        phi1.append(N1index[i])
    phi2=[]
    for z in range(len(y)):
        if(y[z]==16):
            y[z]=15
    for j in y:
        phi2.append(N2index[j])

    index=np.random.permutation(M)
    x=np.ceil(index[0:L]/M2)             
    x=x.astype(int)
    y=index[0:L]-(M2*(x-1))
    psi1=[]
    for d in x:
        psi1.append(M1index[d])
    psi2=[]
    for e in y:
        psi2.append(M2index[e])

    alpha=np.zeros((L,1),dtype=complex)
    alpha[0:L] = np.random.normal(loc=0,scale=1,size=(L,1)) + 1j*np.random.normal(loc=0,scale=1,size=(L,1)) / math.sqrt(2)        #gaussian distribution

    for l in range(L):
        a1=(1/math.sqrt(N1))*np.exp((-1j)*2*np.pi*(np.arange(0,N1,1).reshape(N1,1))*d*phi1[l])
        a2=(1/math.sqrt(N2))*np.exp((-1j)*2*np.pi*(np.arange(0,N2,1).reshape(N2,1))*d*phi2[l])
        a=np.kron(a1,a2)
        b1=(1/math.sqrt(M1))*np.exp((-1j)*2*np.pi*(np.arange(0,M1,1).reshape(M1,1))*d*psi1[l])
        b2=(1/math.sqrt(M2))*np.exp((-1j)*2*np.pi*(np.arange(0,M2,1).reshape(M2,1))*d*psi1[1])
        b=np.kron(b1,b2)
        G = G + alpha[l]*np.matmul(b,a.T)

    G = np.sqrt((M*N)/L)*G
    return G