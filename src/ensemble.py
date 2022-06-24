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
        
        self.epsilon = 1e-3
        
        self.gamma = self.gamma0
        
        self.nconv = self.niter
                
        self.verbose = False
        
        return

    def iterate(self,gamma=None):

        #Check whether input is given for gamma
        if gamma == None:
            #No input, use reference value as starting point
            calibrate = True
            if self.verbose:
                print('Calibrating gamma values')
        else:
            #Use input value for gamma
            self.gamma = gamma
            calibrate = False
            if self.verbose:
                print('Using input gamma values')
                        
        if self.verbose:
            if self.usefanom: 
                print('Using exponential fit through tanom')
            else:
                print('Using raw data of tanom')
                
        self.get_nofeedback(calibrate)
        
        for n in range(1,self.niter+1):
            self.one_iter(n)
            check = np.sum(self.SLR[n,-1,:]-self.SLR[n-1,-1,:])
            if n>1 and np.abs(check)<self.epsilon:
                self.nconv = n
                #print('convergence at n=',self.nconv,100*np.sum(self.SLR[n-1:n+1,-1,:],axis=1),end='                                         \n')
                break
            elif n==self.niter:
                self.nconv = 0
                print('no convergence in iteration, using n=0',100*np.sum(self.SLR[n-1:n+1,-1,:],axis=1),end='                                    \n')
        return
    
    def get_nofeedback(self,calibrate):
        self.TMP = np.zeros((self.niter+1,len(self.ds.time),len(self.ds.basin)))
        self.IML = np.zeros((self.niter+1,len(self.ds.time),len(self.ds.basin)))
        self.SLR = np.zeros((self.niter+1,len(self.ds.time),len(self.ds.basin)))
        
        self.TMP[0,:,:] = self.ds.temp.values
        
        if calibrate:
            self.errfac = 10
            c = 0
            while (self.errfac>1.001 or self.errfac<0.999):
                #Update gamma unless first attempt
                if c>0:
                    self.gamma *= 1./(self.errfac)

                #Do calcuations with current gamma
                self.oce2ice(0)    
                self.oce2slr(0)
                #print(0,c,self.gamma[0],self.errfac)#,self.SLR[0,-1,2]-self.SLR[0,-41,2],np.sum(self.SLR[0,-1,:]-self.SLR[0,-41,:]))

                c+=1
                if c==20:
                    print('no convergence in calibration; using initial guess')
                    self.gamma = self.gamma0
                    break
            #print(self.ssp.values,self.esm.values,self.ism.values,0,c,self.gamma[0],self.errfac[0])
        else:
            self.oce2ice(0)
            self.oce2slr(0)
        
    def one_iter(self,n):
        assert n>0
                
        self.ice2oce(n)
        self.oce2ice(n)
        self.oce2slr(n)
        if self.verbose:
            print(n,100*np.sum(self.SLR[n,-1,:]))
    
    def basalmelt(self,T,bmpexp=2):
        m = self.gamma*(T-self.Tf)*np.abs(T-self.Tf)**(bmpexp-1)
        mref = self.gamma*(self.tref-self.Tf)*np.abs(self.tref-self.Tf)**(bmpexp-1)
        return m-mref[np.newaxis,:]
    
    def oce2ice(self,n):
        #Conversion of ocean temperature anomaly to cumulative ice mass loss
        
        #Define local variable
        TMP = self.TMP[n,:,:]
        assert TMP.shape == self.ds.irf.shape
        
        #Integrate over time
        for t,tt in enumerate(self.ds.rftime):
            if t<1: continue
            F = self.basalmelt(TMP[:t,:])
            dFdt = F[1:t,:]-F[:t-1,:]
            CRF = self.ds.irf[1:t,:].values
            self.IML[n,t,:] = np.sum(CRF[::-1]*dFdt,axis=0)
        return

    def oce2slr(self,n):
        #Conversion of ocean temperature anomaly to cumulative sea level rise
        
        #Define local variables
        TMP = self.TMP[n,:,:]
        assert TMP.shape == self.ds.srf.shape
        
        #Integrate over time
        for t,tt in enumerate(self.ds.rftime):
            if t<1: continue
            F = self.basalmelt(TMP[:t,:])
            dFdt = F[1:t,:]-F[:t-1,:]
            CRF = self.ds.srf[1:t,:].values
            self.SLR[n,t,:] = np.sum(CRF[::-1]*dFdt,axis=0)
            
        #Integrate SLR rate in m/yr to get SLR in m
        self.SLR[n,:,:] = np.cumsum(self.SLR[n,:,:],axis=0)
        
        #Calibrate on total Antarctic:
        self.errfac = np.sum(self.SLR[n,-1,:]-self.SLR[n,-41,:])/self.SLRtarget
        #calibrate on Amundsen:
        #self.errfac = (self.SLR[n,-1,2]-self.SLR[n,-41,2])/.0096
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
        #alpha = [.55,.2,.4,.45,.45]
        alpha = [1.,1.,1.,1.,1.]
        for t,tt in enumerate(self.ds.rftime):
            if t<1: continue
            for e,ex in enumerate(self.ds.exp):
                #Index e now refers to source region of ice mass loss
                for b,bas in enumerate(self.ds.basin):
                    #Time derivative of IML in Gt/yr^2
                    F = IML[1:t,e]-IML[:t-1,e]
                    #Scale for non-linear RF
                    F = np.sign(F)*np.abs(F)**alpha[b]
                    #ORF is time derivative of tanom / fanom in K/yr / Gt/yr
                    if self.usefanom: 
                        #Use exponential fit through tanom
                        CRF = (self.ds.fanom[1:t,e,b].values-self.ds.fanom[:t-1,e,b].values)/(self.pert**alpha[b])
                    else: 
                        #Use raw tanom
                        CRF = (self.ds.tanom[1:t,e,b].values-self.ds.tanom[:t-1,e,b].values)/(self.pert**alpha[b])
                    TMP[t,b] += np.sum(CRF[::-1]*F) #Sum over exp (= source region of ice mass loss)
        self.TMP[n,:,:] = self.TMP[0,:,:] + TMP#Add anomaly to temperature without feedback
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
                
        self.savename = f'../data/ensemble_lin_{self.year0}.nc'
        
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
        c = 0 #Counter to show progress
        self.slr_nf = np.zeros((len(self.ssp),len(self.esm),len(self.ism),len(self.time))) #No feedback
        self.slr_wf = np.zeros((len(self.ssp),len(self.esm),len(self.ism),len(self.time))) #With feedback
        self.gamma = np.zeros((len(self.esm),len(self.ism)))
        
        for e,esm in enumerate(self.esm.values):
            for i,ism in enumerate(self.ism.values):
                #Calibrate to get gamma
                cens = EnsembleMember(self.ad,ism=ism,esm=esm,ssp='245',year0=1871)
                cens.get_nofeedback(calibrate=True)
                self.gamma[e,i] = cens.gamma
                for s,ssp in enumerate(self.ssp.values):
                    #Print progress
                    c+=1 
                    print(f'Computing esm {e+1} of {len(self.esm)} | ism {i+1} of {len(self.ism)} | ssp {s+1} of {len(self.ssp)} | Progress: {100*c/(len(self.ssp)*len(self.esm)*len(self.ism)):.0f}% ',end='           \r')
                    
                    #Calculation without recalibration
                    ens = EnsembleMember(self.ad,ism=ism,esm=esm,ssp=ssp,year0=self.year0)
                    ens.niter = self.niter
                    ens.iterate(gamma=self.gamma[e,i])
                    self.slr_nf[s,e,i,:] = np.sum(ens.SLR[0,:,:],axis=1)
                    self.slr_wf[s,e,i,:] = np.sum(ens.SLR[ens.nconv,:,:],axis=1)
                print(esm,ism,f'{self.gamma[e,i]:3.2f}',self.slr_wf[:,e,i,-1]/self.slr_nf[:,e,i,-1],end='                           \n')
        self.save()
        return
    
    def save(self):  
        self.slr_nf = xr.DataArray(self.slr_nf,dims=('ssp','esm','ism','time'),coords={'ssp':self.ssp,'esm':self.esm,'ism':self.ism,'time':self.time},attrs={'unit':'m','long_name':'Cumulative sea level rise without feedback'})
        self.slr_wf = xr.DataArray(self.slr_wf,dims=('ssp','esm','ism','time'),coords={'ssp':self.ssp,'esm':self.esm,'ism':self.ism,'time':self.time},attrs={'unit':'m','long_name':'Cumulative sea level rise with feedback'})
        self.gamma = xr.DataArray(self.gamma,dims=('esm','ism'),coords={'esm':self.esm,'ism':self.ism},attrs={'unit':'m/yr/K^2','long_name':'Basal melt parameter'})
        
        self.ds = xr.Dataset({'slr_nf':self.slr_nf,'slr_wf':self.slr_wf,'gamma':self.gamma})
        self.ds.to_netcdf(self.savename,mode='w')
        self.ds.close()
        print('Saved',self.savename,'                                           ')