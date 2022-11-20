from matplotlib import pyplot as plt
import numpy as np
import sys

sys.path.insert(1, '/home/nikola/test/osv')
from vaja_1.script2 import loadImage, displayImage

if __name__ == '__main__':
    orig_size = [256,512]
    pxDim=[2,1]
    I = loadImage('/home/nikola/test/osv/vaja_6/lena-256x512-08bit.raw',orig_size,np.uint8)
    displayImage(I,'Orginalna slika',iGridX=[0,511],iGridY=[0,511])

def getRadialValue(iXY,iCP):
    K=iCP.shape[0]
    oValue=np.zeros(K)
    x_i,y_i = iXY
    for k in range(K):
        x_k,y_k=iCP[k]
        r=np.sqrt((x_i-x_k)**2+(y_i-y_k)**2)
        if r>0:
            oValue[k]=-r**2*np.log(r)
    return oValue

def getParameters ( iType , ** kwargs ) :

    if iType == 'affine':
        Tk=np.array([[kwargs['scale'][0],0,0],[0,kwargs['scale'][1],0],[0,0,1]])
        Tt=np.array([[1,0,kwargs['trans'][0]],[0,1,kwargs['trans'][1]],[0,0,1]])
        phi=kwargs['rot']*np.pi/180
        Tr=np.array([[np.cos(phi),-np.sin(phi),0],[np.sin(phi),np.cos(phi),0],[0,0,1]])
        Tg=np.array([[1,kwargs['shear'][0],0],[kwargs['shear'][1],1,0],[0,0,1]])

        oP=Tg @ Tr @ Tt @ Tk
    elif iType=='radial':
        K=kwargs['orig_pts'].shape[0]
        UU=np.zeros((K,K))
        coeff_matrix = np.zeros((K,2),dtype=float) #prvi stolpec vse alfe, drugi vsi bete

        for k in range(K):
            radial_values = getRadialValue(kwargs['orig_pts'][k],kwargs['orig_pts'])
            UU[k,:]=radial_values
        UU_inv = np.linalg.inv(UU)
        alphas = UU_inv@kwargs['mapped_pts'][:,0]
        betas = UU_inv@kwargs['mapped_pts'][:,1]
        coeff_matrix[:,0]=alphas
        coeff_matrix[:,1]=betas
        oP={'pts':kwargs['orig_pts'],'coef':coeff_matrix}


    return oP

if __name__ == '__main__':
    Taffine=getParameters('affine',rot=30,scale=[1,1],trans=[0,0],shear=[0,0])
    XY=np.array([[0,0],[511,0],[0,511],[511,511]])
    UV = np.array([[0,0],[511,0],[0,511],[255,255]])
    P_radial=getParameters(iType='radial',orig_pts = XY,mapped_pts=UV)
    print(P_radial)
    print(Taffine)

def transformImage ( iType , iImage , iDim , iP , iBgr =0 , iInterp =0) :
    Y,X = iImage.shape
    oImage = np.ones((Y,X),dtype=float)*iBgr
    for y in range(Y):
        for x in range(X):
            pt=np.array([x,y])*iDim
            if iType == 'affine':
                pt_1 = np.append(pt,1)
                pt_1=iP @ pt_1
                pt=pt_1[:2]
            elif iType=='radial':
                U=getRadialValue(pt,iP['pts'])
                u=U@iP['coef'][:,0]
                v=U@iP['coef'][:,1]
                pt=np.array([u,v])
            pt=pt/iDim    
            if iInterp==0:
                px=np.round(pt).astype(int)
                if px[0] >= 0 and px[0]<X and px[1]>=0 and px[1]<Y:
                    s = iImage[px[1],px[0]] 
            oImage[y,x]=s
    return oImage

if __name__ == '__main__':
        tI=transformImage(iType='affine',iImage=I,iDim=pxDim,iP=np.linalg.inv(Taffine),iBgr=63)
        displayImage(tI,'Affina preslikana slika',iGridX=[0,511],iGridY=[0,511])

    