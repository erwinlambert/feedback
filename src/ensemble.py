import numpy as np
import xarray as xr
from scipy.optimize import curve_fit


from constants import Constants

class EnsembleMember(Constants):
    """ 
    Input: ad: AllData class
    
    """
    def __init__(self,ad,ism,esm,ssp,year0):
        #Read constants
        Constants.__init__(self)
        
        self.usefanom = True
        self.year0 = year0
        
        #Read input
        self.ad = ad
        self.ds = ad.ds.isel(exp=slice(0,5),basin=slice(0,5))
        self.ds = self.ds.sel(ism=ism,esm=esm,ssp=ssp,time=slice(self.year0,self.year0+self.ad.nyears-1))
        
        self.basin = self.ds.basin
        self.exp = self.ds.exp
        self.ism = self.ds.ism
        self.esm = self.ds.esm
        self.ssp = self.ds.ssp
        self.tref = self.ds.tref.values
        self.time = self.ds.time
        
        #self.gamma = self.gamma0+np.zeros(self.niter+1)
        
        self.verbose = False
        
        return

    def iterate(self,gamma=np.array([None,None])):
        if gamma.any()==None:
            self.gamma = self.gamma0+np.zeros(self.niter+1)
            calibrate = True
            #print('Calibrating gamma values')
        else:
            self.gamma = gamma
            calibrate = False
            #print('Using input gamma values')
        
        assert len(self.gamma)==self.niter+1
        
        if self.verbose:
            if self.usefanom:
                print('Using exponential fit through tanom')
            else:
                print('Using raw data of tanom')
                
        self.get_nofeedback(calibrate)
        
        for n in range(1,self.niter+1):
            self.one_iter(n,calibrate)
            
        if self.verbose: print(f'Finished {self.niter} iteration')
        
        return
    
    def get_nofeedback(self,calibrate):
        self.TMP = np.zeros((self.niter+1,len(self.ds.time),len(self.ds.basin)))
        self.IML = np.zeros((self.niter+1,len(self.ds.time),len(self.ds.basin)))
        self.SLR = np.zeros((self.niter+1,len(self.ds.time),len(self.ds.basin)))
        
        self.TMP[0,:,:] = self.ds.temp.values
        
        if calibrate:
            self.errfac = 10
            c = 0
            while (self.errfac>1.01 or self.errfac<0.99):
                #Update gamma unless first attempt
                if c>0:
                    self.gamma[0] *= 1./(self.errfac)
                #print(0,c,self.gamma[0],self.errfac[0])

                #Do calcuations with current gamma
                self.oce2ice(0)    
                self.oce2slr(0)

                c+=1
                if c==20:
                    print('no convergence in calibration; using initial value')
                    self.gamma[0] = self.gamma0
                    break
            #print(self.ssp.values,self.esm.values,self.ism.values,0,c,self.gamma[0],self.errfac[0])
        else:
            self.oce2ice(0)
            self.oce2slr(0)
        
    def one_iter(self,n,calibrate):
        assert n>0
        
        if calibrate:
            self.errfac = 10
            c = 0
            while (self.errfac>1.01 or self.errfac<0.99):
                if c>0:
                    self.gamma[n] *= 1./(self.errfac)
                #print(n,c,self.gamma[n],self.errfac[n])

                #Do calculations with current gamma
                self.ice2oce(n)
                self.oce2ice(n)
                self.oce2slr(n)

                c+=1
                if c==20:
                    print('no convergence in calibration; using initial value')
                    self.gamma[n] = self.gamma0
                    break

            #print(self.ssp.values,self.esm.values,self.ism.values,n,c,self.gamma[n],self.errfac[n])
        else:
            self.ice2oce(n)
            self.oce2ice(n)
            self.oce2slr(n)            
    
    def basalmelt(self,T,gamma,bmpexp=2):
        #return self.gamma[self.bmp][np.newaxis,:]*self.spy*self.K**self.bmpexp[self.bmp]*(T-self.Tf)*np.abs(T-self.Tf)**(self.bmpexp[self.bmp]-1)
        #return self.gamma[self.bmp][np.newaxis,:]*(T-self.Tf)*np.abs(T-self.Tf)**(self.bmpexp[self.bmp]-1)
        #return self.gamma[self.bmp][np.newaxis,:]*(T-self.tref[np.newaxis,:])*np.abs(self.Trean[np.newaxis,:]-self.Tf)**(self.bmpexp[self.bmp]-1)
        m = gamma*(T-self.Tf)*np.abs(T-self.Tf)**(bmpexp-1)
        mref = gamma*(self.tref-self.Tf)*np.abs(self.tref-self.Tf)**(bmpexp-1)
        return m-mref[np.newaxis,:]
    
    def oce2ice(self,n):
        #Conversion of ocean temperature anomaly to cumulative ice mass loss
        
        #Define local variable
        TMP = self.TMP[n,:,:]
        assert TMP.shape == self.ds.irf.shape
        
        #Integrate over time
        for t,tt in enumerate(self.ds.rftime):
            if t<1: continue
            F = self.basalmelt(TMP[:t,:],self.gamma[n])
            CRF = self.ds.irf[:t,:].values
            self.IML[n,t,:] = np.sum(CRF[::-1]*F,axis=0)
        return

    def oce2slr(self,n):
        #Conversion of ocean temperature anomaly to cumulative sea level rise
        
        #Define local variables
        TMP = self.TMP[n,:,:]
        assert TMP.shape == self.ds.srf.shape
        
        #Integrate over time
        for t,tt in enumerate(self.ds.rftime):
            if t<1: continue
            F = self.basalmelt(TMP[:t,:],self.gamma[n])
            CRF = self.ds.srf[:t,:].values
            self.SLR[n,t,:] = np.sum(CRF[::-1]*F,axis=0)
        
        self.errfac = np.sum(self.SLR[n,-1,:]-self.SLR[n,-41,:])/self.SLRtarget
        return
    
    def ice2oce(self,n):
        #Conversion of ice mass loss rate to temperature anomaly
        assert n>0
        
        #Define local variables
        IML = self.IML[n-1,:,:]
        assert IML.shape == self.ds.tanom.shape[:2]
        assert self.ds.tanom.shape[1] == self.ds.tanom.shape[2]
        TMP = 0.*IML
        
        #Integrate over time
        alpha = .5
        for t,tt in enumerate(self.ds.rftime):
            if t<1: continue
            for e,ex in enumerate(self.ds.exp):
                #Index e now refers to source region of ice mass loss
                #Time derivative of IML in Gt/yr
                F = IML[1:t,e]-IML[:t-1,e]
                #Scale for non-linear RF
                F = np.sign(F)*np.abs(F)**alpha
                for b,bas in enumerate(self.ds.basin):
                    #ORF is time derivative of tanom / fanom in K/yr / Gt/yr
                    if self.usefanom: 
                        #Use exponential fit through tanom
                        CRF = (self.ds.fanom[1:t,e,b].values-self.ds.fanom[:t-1,e,b].values)/(self.pert**alpha)
                    else: 
                        #Use raw tanom
                        CRF = (self.ds.tanom[1:t,e,b].values-self.ds.tanom[:t-1,e,b].values)/(self.pert**alpha)
                    TMP[t,b] += np.sum(CRF[::-1]*F) #Sum over exp (= source region of ice mass loss)
        self.TMP[n,:,:] = self.TMP[n-1,:,:] + TMP #Add anomaly to temperature of previous iteration
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
        
        self.year0 = 1951
        ds = ds.sel(time=slice(self.year0,self.year0+150-1))
        
        self.basin = ds.basin
        self.exp = ds.exp
        self.ism = ds.ism
        self.esm = ds.esm
        self.ssp = ds.ssp
        self.time = ds.time
        
        self.savename = f'../data/ensemble_{self.year0}.nc'
        
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
        self.slr_nf = np.zeros((len(self.ssp),len(self.esm),len(self.ism),len(self.time)))
        self.slr_wf = np.zeros((len(self.ssp),len(self.esm),len(self.ism),len(self.time)))
        self.gamma = np.zeros((len(self.esm),len(self.ism),self.niter+1))
        
        for e,esm in enumerate(self.esm):
            for i,ism in enumerate(self.ism):
                #Calibrate to get gamma
                cens = EnsembleMember(self.ad,ism=ism,esm=esm,ssp='245',year0=1871)
                cens.niter = self.niter
                cens.iterate()
                self.gamma[e,i,:] = cens.gamma
                #self.gamma[e,i,1] = self.gamma[e,i,0] #REMOVE! Or make optional
                #print(self.gamma[e,i,:])
                for s,ssp in enumerate(self.ssp):
                    c+=1
                    ens = EnsembleMember(self.ad,ism=ism,esm=esm,ssp=ssp,year0=self.year0)
                    ens.niter = self.niter
                    ens.iterate(gamma=self.gamma[e,i,:])
                    self.slr_nf[s,e,i,:] = np.sum(ens.SLR[0,:,:],axis=1)
                    self.slr_wf[s,e,i,:] = np.sum(ens.SLR[-1,:,:],axis=1)
                    print(f'Got esm {e+1} of {len(self.esm)} | ism {i+1} of {len(self.ism)} | ssp {s+1} of {len(self.ssp)} | Progress: {100*c/(len(self.ssp)*len(self.esm)*len(self.ism)):.0f}% ',end='           \r')
        self.save()
        return
    
    def save(self):  
        self.slr_nf = xr.DataArray(self.slr_nf,dims=('ssp','esm','ism','time'),coords={'ssp':self.ssp,'esm':self.esm,'ism':self.ism,'time':self.time},attrs={'unit':'m','long_name':'Cumulative sea level rise without feedback'})
        self.slr_wf = xr.DataArray(self.slr_wf,dims=('ssp','esm','ism','time'),coords={'ssp':self.ssp,'esm':self.esm,'ism':self.ism,'time':self.time},attrs={'unit':'m','long_name':'Cumulative sea level rise with feedback'})
        self.gamma_nf = xr.DataArray(self.gamma[:,:,0],dims=('esm','ism'),coords={'esm':self.esm,'ism':self.ism},attrs={'unit':'m/yr/K^2','long_name':'Basal melt sensitivity without feedback'})
        self.gamma_wf = xr.DataArray(self.gamma[:,:,-1],dims=('esm','ism'),coords={'esm':self.esm,'ism':self.ism},attrs={'unit':'m/yr/K^2','long_name':'Basal melt sensitivity with feedback'})    
        
        self.ds = xr.Dataset({'slr_nf':self.slr_nf,'slr_wf':self.slr_wf,'gamma_nf':self.gamma_nf,'gamma_wf':self.gamma_wf})
        self.ds.to_netcdf(self.savename,mode='w')
        self.ds.close()
        print('Saved',self.savename,'                                           ')