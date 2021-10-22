import numpy as np
import xarray as xr

from constants import Constants

class EnsembleMember(Constants):
    """ 
    Input: ad: AllData class
    
    """
    def __init__(self,ad,ism,esm,ssp,bmp):
        #Read constants
        Constants.__init__(self)
        
        #Read input
        self.ds = ad.ds.isel(exp=slice(0,5),basin=slice(0,5))
        self.ds = self.ds.sel(ism=ism,esm=esm,ssp=ssp)
        
        self.basin = self.ds.basin
        self.exp = self.ds.exp
        self.ism = self.ds.ism
        self.esm = self.ds.esm
        self.ssp = self.ds.ssp
        
        self.bmp = bmp
        
        self.verbose = False
        
        return

    def iterate(self):
        self.get_nofeedback()
        for n in range(1,self.niter+1):
            self.one_iter(n)
        if self.verbose: print(f'Finished {self.niter} iteration')
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
    
    
class FullEnsemble(Constants):
    """ 
    Input: ad: AllData class
    
    """
    def __init__(self,ad):
        #Read constants
        Constants.__init__(self)
        
        #Read input
        self.ad = ad
        ds = ad.ds.isel(exp=slice(0,5),basin=slice(0,5))
        
        self.basin = ds.basin
        self.exp = ds.exp
        self.ism = ds.ism
        self.esm = ds.esm
        self.ssp = ds.ssp
        self.time = ds.time
        
        self.savename = f'../data/ensemble{self.ad.option}_{self.ad.year0}.nc'
        
        return
    
    def gather(self,force_update=False):
        if force_update:
            print('Doing a forced update of ensemble calculation')
            self.compute()
        else:
            try:
                print('Reading old saved ensemble')
                ds = xr.open_dataset(self.savename)
                self.slr_nf = ds['slr_nf']
                self.slr_wf = ds['slr_wf']
                ds.close()
            except:
                print('New input parameters, calculating new ensemble')
                self.compute()
        return
    
    def compute(self):
        c = 0
        self.slr_nf = np.zeros((len(self.bmps),len(self.ssp),len(self.esm),len(self.ism),len(self.time)))
        self.slr_wf = np.zeros((len(self.bmps),len(self.ssp),len(self.esm),len(self.ism),len(self.time)))
        for b,bmp in enumerate(self.bmps):
            for s,ssp in enumerate(self.ssp):
                for e,esm in enumerate(self.esm):
                    for i,ism in enumerate(self.ism):
                        c+=1
                        ens = EnsembleMember(self.ad,ism=ism,esm=esm,ssp=ssp,bmp=bmp)
                        ens.iterate()
                        self.slr_nf[b,s,e,i,:] = np.sum(ens.SLR[0,:,:],axis=1)
                        self.slr_wf[b,s,e,i,:] = np.sum(ens.SLR[-1,:,:],axis=1)
                        print(f'Got bmp {b+1} of {len(self.bmps)} | ssp {s+1} of {len(self.ssp)} | esm {e+1} of {len(self.esm)} | ism {i+1} of {len(self.ism)} | Progress: {100*c/(len(self.ssp)*len(self.esm)*len(self.ism)*len(self.bmps)):.0f}% ',end='           \r')
        self.save()
        return
    
    def save(self):  
        self.slr_nf = xr.DataArray(self.slr_nf,dims=('bmp','ssp','esm','ism','time'),coords={'bmp':self.bmps,'ssp':self.ssp,'esm':self.esm,'ism':self.ism,'time':self.time},attrs={'unit':'m','long_name':'Cumulative sea level rise without feedback'})
        self.slr_wf = xr.DataArray(self.slr_wf,dims=('bmp','ssp','esm','ism','time'),coords={'bmp':self.bmps,'ssp':self.ssp,'esm':self.esm,'ism':self.ism,'time':self.time},attrs={'unit':'m','long_name':'Cumulative sea level rise with feedback'})

        self.ds = xr.Dataset({'slr_nf':self.slr_nf,'slr_wf':self.slr_wf})
        self.ds.to_netcdf(self.savename,mode='w')
        self.ds.close()
        print('Saved',self.savename,'                                           ')