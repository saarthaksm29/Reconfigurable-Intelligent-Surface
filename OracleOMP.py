from dsomp3 import Full_Supports, Phi_s
import numpy as np
import math

def OracleOMP(Y,H,Phi,M,N,K,L1,L2):
    Hhat=np.zeros((K,N,M),dtype=complex)
    Full_Supports=np.zeros((K,L1,L2),dtype=complex)

    for k in range(K):
        HK=H[k,:,:]
        YK=Y[k,:,:]
        s=np.sum(np.square(np.abs(HK)),axis=0)
        order=s.argsort()[::-1]
        row=order[0:L1]
        row=row

        for r in range(L1):
            h=HK[:,row[r]]
            y=YK[:,row[r]]
            a=np.square(np.abs(h))
            order=a.argsort()[::-1]
            support=order[0:L2]
            Phi_s=Phi[:,support]
            hhat=np.zeros(N,1)
            hhat[support,:]=np.matmul(np.linalg.inv(np.matmul(Phi_s.T,Phi_s)),np.matmul(Phi_s.T,y))
            Hhat[k,:,row[r]]=hhat
            Full_Supports[k,:,r]=support
    
    return [Hhat,row,Full_Supports]