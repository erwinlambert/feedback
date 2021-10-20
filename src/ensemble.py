import numpy as np
import xarray as xr

from constants import Constants

class EnsembleMember(Constants):
    """ Data and attributes for a single run 
    
    """
    def __init__(self,ds,ism,esm,ssp):
        #Read constants
        Constants.__init__(self)
        
        #Read input
        self.ds = ds.isel(exp=slice(0,5),basin=slice(0,5))
        self.ds = self.ds.sel(ism=ism,esm=esm,ssp=ssp)
        
        self.basin = self.ds.basin
        self.exp = self.ds.exp
        self.ism = self.ds.ism
        self.esm = self.ds.esm
        self.ssp = self.ds.ssp
        
        self.niter = 1
        self.bmp = 'lin'
        self.Tf = -1.7
        
        return

    def iterate(self):
        self.get_nofeedback()
        for n in range(1,self.niter+1):
            self.one_iter(n)
        return
    
    def get_nofeedback(self):
        self.TMP = np.zeros((self.niter+1,len(self.ds.time),len(self.ds.basin)))
        self.IML = np.zeros((self.niter+1,len(self.ds.time),len(self.ds.basin)))
        self.SLR = np.zeros((self.niter+1,len(self.ds.time),len(self.ds.basin)))
        self.TMP[0,:,:] = self.ds.temp.values
        self.oce2ice(0)
        self.oce2slr(0)

    def one_iter(self,n):
        assert n>0
        self.ice2oce(n)
        self.oce2ice(n)
        self.oce2slr(n)
        return
    
    def basalmelt(self,T):
        return self.gamma[self.bmp][np.newaxis,:]*self.spy*self.K**self.bmpexp[self.bmp]*(T-self.Tf)*np.abs(T-self.Tf)**(self.bmpexp[self.bmp]-1)
    
    def oce2ice(self,n):
        #Conversion of ocean temperature anomaly to ice mass loss rate
        
        TMP = self.TMP[n,:,:]
        assert TMP.shape == self.ds.irf.shape
        IML = 0.*TMP
        for t,tt in enumerate(self.ds.rftime):
            if t<1: continue
            dFdt = self.basalmelt(TMP[1:t,:])-self.basalmelt(TMP[:t-1,:])
            CRF  = self.ds.irf[1:t,:].values
            for b,bas in enumerate(self.ds.basin):
                IML[t,b] = np.sum(CRF[::-1,b]*dFdt[:,b])
        self.IML[n,:,:] = IML
        return

    def oce2slr(self,n):
        #Conversion of ocean temperature anomaly to cumulative sea level rise
        TMP = self.TMP[n,:,:]
        assert TMP.shape == self.ds.srf.shape
        SLR = 0.*TMP
        for t,tt in enumerate(self.ds.rftime):
            if t<1: continue
            dFdt = self.basalmelt(TMP[1:t,:])-self.basalmelt(TMP[:t-1,:])
            CRF  = self.ds.srf[1:t,:]         
            for b,bas in enumerate(self.ds.basin):
                SLR[t,b] = np.sum(CRF[::-1,b]*dFdt[:,b])
        self.SLR[n,:,:] = np.cumsum(SLR,axis=0)
        return
    
    def ice2oce(self,n):
        #Conversion of ice mass loss rate to temperature anomaly
        assert n>0
        IML = self.IML[n-1,:,:]
        assert IML.shape == self.ds.tanom.shape[:2]
        assert self.ds.tanom.shape[1] == self.ds.tanom.shape[2]
        TMP = 0.*IML
        for t,tt in enumerate(self.ds.rftime):
            if t<1: continue
            for e,ex in enumerate(self.ds.exp):
                dFdt = IML[1:t,e]-IML[:t-1,e]
                for b,bas in enumerate(self.ds.basin):
                    CRF = self.ds.tanom[1:t,e,b].values/self.pert
                    TMP[t,b] += np.sum(CRF[::-1]*dFdt) #Sum over basins
        self.TMP[n,:,:] = self.TMP[n-1,:,:] + TMP
        return