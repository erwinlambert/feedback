import numpy as np
import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 1.
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.subplot.wspace'] = .25
mpl.rcParams['figure.subplot.left'] = .1
mpl.rcParams['figure.subplot.right'] = .95
mpl.rcParams['figure.figsize'] = (7,5)

basin = ['East Ant.','Ross','Amundsen','Weddell','Peninsula']
exp = ['EAIS','ROSS','AMUN','WEDD','PENS','TOTL','HALF','DOUB']

bcol = {}
bcol['East Ant.'] = 'tab:blue'
bcol['Ross']      = 'tab:orange'
bcol['Amundsen']  = 'tab:red'
bcol['Weddell']   = 'tab:purple'
bcol['Peninsula'] = 'tab:green'

scol = {}
scol['119'] = (0/255,173/255,207/255)
scol['126'] = (23/255,60/255,102/255)
scol['245'] = (247/255,148/255,32/255)
scol['370'] = (231/255,29/255,37/255)
scol['585'] = (149/255,27/255,30/255)


spy = 3600*24*365.25 #Seconds per year

bmps = ['lin','quad']
gamma = {} #Gamma value [m/s] from Favier et al 2019
gamma['lin'] = 2e-5
gamma['quad'] = 36e-5
bmpexp = {} #Exponent in basal melt equation
bmpexp['lin'] = 1
bmpexp['quad'] = 2

K = 1028*3947/(917*3.34e5)

def basalmelt(T,bmp='lin',Tf=-1.7):
    return gamma[bmp]*spy*K**bmpexp[bmp]*(T-Tf)*np.abs(T-Tf)**(bmpexp[bmp]-1)


def oce2ice(TMP,IRF,bmp='lin'):
    #Input:
    #TMP: 5 temperature time series
    #IRF: Ice response function
    #bmp: basal melt parametrisation
    #Output:
    #IML: Ice mass loss for 5 regions [Gt/yr]
    assert TMP.shape == IRF.shape
    IML = 0.*TMP
    for t,tt in enumerate(IRF.rftime):
        if t==0: continue
        for b,bas in enumerate(IRF.basin):
            dFdt = basalmelt(TMP[1:t,b],bmp=bmp)-basalmelt(TMP[:t-1,b],bmp=bmp)
            CRF = IRF[1:t,b].values
            IML[t,b] = np.sum(CRF[::-1]*dFdt)
    return IML

def oce2slr(TMP,SRF,bmp='lin'):
    #Input:
    #TMP: 5 temperature time series
    #SRF: Sealevel response function
    #bm: basal melt sensitivity [m/yr /degC]
    #Output:
    #SLR: Cumulative sea level rise for 5 regions [m]
    assert TMP.shape == SRF.shape
    SLR = 0.*TMP
    for t,tt in enumerate(SRF.rftime):
        if t==0: continue
        for b,bas in enumerate(SRF.basin):
            dFdt = basalmelt(TMP[1:t,b],bmp=bmp)-basalmelt(TMP[:t-1,b],bmp=bmp)
            CRF = SRF[1:t,b].values
            SLR[t,b] = np.sum(CRF[::-1]*dFdt)
    return np.cumsum(SLR,axis=0)

def ice2oce(IML,ORF):
    #Input:
    #IML: 5 ice mass loss time series
    #ORF: Ocean response function
    #Output:
    #TMP: Temperature anomaly for 5 regions [degC]
    assert IML.shape == ORF.shape[:2]
    assert ORF.shape[1] == ORF.shape[2]
    TMP = 0.*IML
    for t,tt in enumerate(ORF.rftime):
        if t==0: continue
        for e,ex in enumerate(ORF.exp):
            dFdt = IML[1:t,e]-IML[:t-1,e]
            for b,bas in enumerate(ORF.basin):
                CRF = ORF[1:t,e,b].values
                TMP[t,b] += np.sum(CRF[::-1]*dFdt)
    return TMP

def iterate(ds,ism,esm,ssp,bmp='lin',niter=1):
    niter += 1
    TMP = np.zeros((niter,len(ds.time),len(ds.basin)))
    IML = np.zeros((niter,len(ds.time),len(ds.basin)))
    SLR = np.zeros((niter,len(ds.time),len(ds.basin)))
    TMP[0,:,:] = ds.temp.sel(esm=esm,ssp=ssp).values
    IML[0,:,:] = oce2ice(TMP[0,:,:],ds.irf.sel(ism=ism),bmp=bmp)
    SLR[0,:,:] = oce2slr(TMP[0,:,:],ds.srf.sel(ism=ism),bmp=bmp)
    for n in range(1,niter):
        TMP[n,:,:] = ds.temp.sel(esm=esm,ssp=ssp).values + ice2oce(IML[n-1,:,:],ds.orf)
        IML[n,:,:] = oce2ice(TMP[n,:,:],ds.irf.sel(ism=ism),bmp=bmp)
        SLR[n,:,:] = oce2slr(TMP[n,:,:],ds.srf.sel(ism=ism),bmp=bmp)
    return TMP,IML,SLR