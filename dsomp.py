import numpy as np
import math

from numpy.core.defchararray import index

def DS_OMP(Y,Phi,M,N,K,L1,L2,L2c):
    
    Hhat=np.zeros((K,N,M),dtype=complex)
    T=np.size(Phi,0)

    e=0
    for k in range(K):
        Yk = Y[k,:,:]
        s = np.absolute(Yk)
        s = np.square(s)
        s = np.sum(s,axis=0)
        e = e + s
    index=e.argsort()[::-1]
    column=index[0:L1]
    Full_Supports = np.zeros((K,L1,L2),dtype=complex)
    ############ Finding the common row and common column support ################
    for c in range(L1):
        Yc=Y[:,:,column[c]]
        Yc=np.reshape(Yc,(T,K))
        Rc=Yc
        rowc=[]
        Hchat=np.zeros((N,K),dtype=complex)
        ########## Double Structured Orthogonal Matching Pursuit ############
        if L2c>0:
            FullRow=np.zeros((L2,K),dtype=complex)
            for k in range(K):
                k
                yc=Yc[:,k]  #review
                rc=Rc[:,k]  #review
                row=[]
                for r in range(L2):
                    e=np.square(np.absolute(np.matmul(Phi.T,rc)))
                    index=np.argmax(e) 
                    row.append(index)
                    Phi_s=Phi[:,row]        
                    hchat=np.zeros((N,1),dtype=complex)
                    hchat[row,:]=np.linalg.inv(Phi_s.T*Phi_s)*Phi_s.T*yc
                    rc=yc-np.matmul(Phi,hchat)

                FullRow[:,k]=row
            
            count=np.zeros((1,N),dtype=complex)
            Fullcount=FullRow[:]
            for i in range(len(Fullcount)):
                count[Fullcount[i]] = count[Fullcount[i]]+1

            index=count.argsort()[::-1]
            rowc=index[0:L2c]

            Phi_s=Phi[:,rowc]
            Hchat=np.zeros((N,K),dtype=complex)
            Hchat[rowc,:]=np.linalg.inv(Phi_s.T*Phi_s)*Phi_s.T*Yc
            Rc=Yc-np.matmul(Phi,Hchat)
            Hhat[:,:,column[c]]=Hchat  

        for k in range(K):
            k
            yc=Yc[:,k]  
            rc=Rc[:,k] 
            row=rowc
            for r in range(L2-L2c):
                e=np.square(np.absolute(Phi.T*rc))
                index=np.argmax(e) 
                row.append(index)
                Phi_s=Phi[:,row]
                hchat=np.zeros((N,1),dtype=complex)
                hchat[row,:]=np.linalg.inv(Phi_s.T*Phi_s)*Phi_s.T*yc
                rc=yc-Phi*hchat
            
            if(L2-L2c>0):
                Hhat[k,:,column[c]]=hchat

            Full_Supports[k,:,c]=row

    return [Hhat,column,Full_Supports]
