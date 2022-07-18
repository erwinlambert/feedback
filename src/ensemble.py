import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


from constants import Constants
       

class EnsembleMember(Constants):
    """ 
    Input: ad: AllData class
    
    """
    def __init__(self,ad,ism,esm,ssp,year0,cal='S',nonlin='cutoff'):
        #Read constants
        Constants.__init__(self)
        
        self.usefanom = True
        self.year0 = year0
        assert self.year0 < 1979
        assert self.year0 > 2017-ad.nyears
        
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
        
        #Calibration time series
        self.cds = ad.ds.isel(exp=slice(0,5),basin=slice(0,5))
        self.cds = self.cds.sel(ism=ism,esm=esm,ssp='245',time=slice(self.year0,self.year0+self.ad.nyears-1))
        
        self.epsilon = 1e-3
        
        self.gamma = np.array([self.gamma0,self.gamma0])
        
        self.nconv = self.niter
        
        self.cal = cal #'S' or 'I'
        self.nonlin = nonlin #'alpha' or 'cutoff'
        
        #Allocate variables for nf (no feedback), wf (with feedback)
        self.TMP = np.zeros((2,len(self.ds.time),len(self.ds.basin)))
        self.IML = np.zeros((2,len(self.ds.time),len(self.ds.basin)))
        self.SLR = np.zeros((2,len(self.ds.time),len(self.ds.basin)))
        self.MLT = np.zeros((2,len(self.ds.time),len(self.ds.basin)))
                
        self.verbose = False
        
        return

    def compute(self,gamma=np.array([None,None]),recal=True):
        assert self.cal in ['S','I']

        #Check whether input is given for gamma with feedback
        if gamma.any() == None:
            #No input, use reference value as starting point
            calibrate = True
            if self.verbose:
                print('Calibrating gamma value')
        else:
            #Use input values for gamma
            assert len(gamma) == 2
            self.gamma = gamma
            calibrate = False
            if self.verbose:
                print('Using input gamma value')
        
        #No feedback
        if calibrate:
            self.calibrate_nf()
            
        if np.isnan(self.gamma[0]):
            #Could not calibrate, abort computation for this ensemble member
            self.TMP[:,:,:] = np.nan
            self.IML[:,:,:] = np.nan
            self.SLR[:,:,:] = np.nan
            self.MLT[:,:,:] = np.nan
            if self.verbose:
                print('no calibration possible without feedback, setting all nan')
            return
        
        self.get_nofeedback()
        if self.verbose:
            print('no feedback',np.array([self.gamma[0],np.sum(self.IML[0,:,:]),100*np.sum(self.SLR[0,-1,:])]))

        #With feedback
        
        if recal:
            if calibrate:
                self.calibrate_wf()
            if np.isnan(self.gamma[1]):
                #Could not calibrate with feedback, set all to nan and abort computation
                self.gamma[0] = np.nan
                self.TMP[:,:,:] = np.nan
                self.IML[:,:,:] = np.nan
                self.SLR[:,:,:] = np.nan
                if self.verbose:
                    print('no calibration possible with feedback, setting all nan')
                return

        self.TMP[1,:,:],self.IML[1,:,:],self.SLR[1,:,:],err = self.get_withfeedback(self.gamma[1],self.ds.temp.values)
        self.MLT[1,:,:] = self.basalmelt(self.TMP[1,:,:],self.gamma[1])
        if self.verbose:
            print('wf with recal',np.array([self.gamma[1],np.sum(self.IML[1,:,:]),100*np.sum(self.SLR[1,-1,:])]))

        return
    
    def calibrate_nf(self):
        #Get gamma no feedback
        
        TMP = self.cds.temp.values
        err = {}
        err['I'] = 10
        err['S'] = 10
        c = 0
        while (err[self.cal]>1.01 or err[self.cal]<0.99):
            #Update gamma unless first attempt
            if c>0:
                self.gamma[0] *= 1./(err[self.cal])

            #Do calcuations with current gamma
            
            IML,err['I'] = self.oce2ice(TMP,self.gamma[0])    
            SLR,err['S'] = self.oce2slr(TMP,self.gamma[0])
            
            #print(0,c,self.gamma[0],self.errfac)#,self.SLR[0,-1,2]-self.SLR[0,-41,2],np.sum(self.SLR[0,-1,:]-self.SLR[0,-41,:]))

            c+=1
            if c==20:
                print('no convergence in calibration no feedback; using initial guess')
                self.gamma[0] = self.gamma0
                break
        if self.gamma[0]<0:
            self.gamma[:] = np.nan
        return
    
    def calibrate_wf(self):
        #Get gamma_wf (calibration with feedback)
        
        if np.isnan(self.gamma[0]):
            self.gamma[1] = np.nan
            return
        
        #First try
        gamma = 1.5#self.gamma0
        TMP,IML,SLR,err = self.get_withfeedback(gamma,self.cds.temp.values)
        errgam = np.array([[gamma,err[self.cal]]])

        #Second try
        gamma = .8
        TMP,IML,SLR,err = self.get_withfeedback(gamma,self.cds.temp.values)
        errgam = np.append(errgam,[[gamma,err[self.cal]]],axis=0)
        
        #Third try
        gamma = .3
        TMP,IML,SLR,err = self.get_withfeedback(gamma,self.cds.temp.values)
        errgam = np.append(errgam,[[gamma,err[self.cal]]],axis=0)

        #Third try
        gamma = .1
        TMP,IML,SLR,err = self.get_withfeedback(gamma,self.cds.temp.values)
        errgam = np.append(errgam,[[gamma,err[self.cal]]],axis=0)
        
        ntries = 20
        for c in range(ntries):
            lasttry = errgam[-1,0]
            lasterr = errgam[-1,1]
            
            if self.verbose:
                print(lasttry,lasterr)
            
            errgam = errgam[errgam[:,0].argsort()]
            amax = np.argmax(errgam[:,1])
            amin = np.argmin(errgam[:,1])
            if self.verbose:
                print(errgam)
            if errgam[amax,1]>1 and errgam[amin,1]<1:
                #Interpolate towards convergence
                xarr = np.arange(np.min(errgam[:,0]),np.max(errgam[:,0]),.001)
                yarr = interp1d(errgam[:,0],errgam[:,1],kind='quadratic')
                nexttry = xarr[np.argmin((yarr(xarr)-1)**2)]
                if self.verbose:
                    print('interpolating, try',nexttry)
                
            elif amax > 0 and amax < errgam.shape[0]-1:
                if amax>1 and amax < errgam.shape[0]-2:
                    #Covered enough, no calibration possible
                    if self.verbose: 
                        print('no calibration possible, gamma = nan')
                else:
                    #Quadratic fit, try to find maximum
                    fit = np.polyfit(errgam[:,0],errgam[:,1],2)
                    nexttry = -fit[1]/(2*fit[0])
                    if self.verbose:
                        print('interpolating quadratic, try',nexttry)
            else:
                #Haven't convered full range yet, extrapolate
                fit = np.polyfit(errgam[:,0],errgam[:,1],1)
                nexttry = (1-fit[1])/fit[0]
                nexttry = lasttry+3*(nexttry-lasttry) #Take a bigger step
                if self.verbose:
                    print('extrapolating, try',nexttry)
                
            if np.abs(nexttry-lasttry)<.005:
                if np.abs(lasterr-1)<.25:
                    if self.verbose:
                        print('calibration finished, gamma = ',nexttry,'last error = ',err[self.cal])
                    self.gamma[1] = nexttry
                    break
                else:
                    if self.verbose:
                        print('no calibration possible, gamma = nan')   
                    self.gamma[1] = np.nan
                    break
                    
            TMP,IML,SLR,err = self.get_withfeedback(nexttry,self.cds.temp.values)
            #print(nexttry,errfac)
            errgam = np.append(errgam,[[nexttry,err[self.cal]]],axis=0)
                    
            if c==ntries-1:
                if self.verbose:
                    print('no calibration possible, gamma = nan')
                self.gamma[1] = np.nan
                break
        return
    
    def get_nofeedback(self):
        #Function to get TMP,IML,SLR based on calibrated gamma with no feedback
        
        self.TMP[0,:,:]      = self.ds.temp.values
        self.IML[0,:,:],errI = self.oce2ice(self.TMP[0,:,:],self.gamma[0])    
        self.SLR[0,:,:],errS = self.oce2slr(self.TMP[0,:,:],self.gamma[0])
        self.MLT[0,:,:]      = self.basalmelt(self.TMP[0,:,:],self.gamma[0])
        
        return
    
    def get_withfeedback(self,gamma,tmp0):
        #Iteration with input gamma
        
        TMP = np.zeros((self.niter+1,len(self.ds.time),len(self.ds.basin)))
        IML = np.zeros((self.niter+1,len(self.ds.time),len(self.ds.basin)))
        SLR = np.zeros((self.niter+1,len(self.ds.time),len(self.ds.basin)))
        
        err = {}
        
        #First calculation
        TMP[0,:,:]          = tmp0
        IML[0,:,:],err['I'] = self.oce2ice(TMP[0,:,:],gamma) 
        SLR[0,:,:],err['S'] = self.oce2slr(TMP[0,:,:],gamma)
        
        #Iterate
        for n in range(1,self.niter+1):
            TMP[n,:,:]          = self.ice2oce(IML[n-1,:,:]) + TMP[0,:,:]
            IML[n,:,:],err['I'] = self.oce2ice(TMP[n,:,:],gamma) 
            SLR[n,:,:],err['S'] = self.oce2slr(TMP[n,:,:],gamma)
            #Check whether SLR converges
            check = np.sum(SLR[n,-1,:])/np.sum(SLR[n-1,-1,:])-1
            #print(gamma,check)
            if n>1 and np.abs(check)<self.epsilon:
                #Succesful convergence
                nconv = n
                break
            elif n==self.niter:
                nconv = 0
                print('no convergence in iteration, using n=0',end='                                                              \n')

        return TMP[nconv,:,:],IML[nconv,:,:],SLR[nconv,:,:],err
           
    def basalmelt(self,T,gamma,bmpexp=2):
        m = gamma*(T-self.Tf)*np.abs(T-self.Tf)**(bmpexp-1)
        mref = gamma*(self.tref-self.Tf)*np.abs(self.tref-self.Tf)**(bmpexp-1)
        return m-mref[np.newaxis,:]
    
    def oce2ice(self,TMP,gamma):
        #Conversion of ocean temperature anomaly to cumulative ice mass loss
        
        #Define local variable
        assert TMP.shape == self.ds.irf.shape
        IML = 0.*TMP
        
        #Integrate over time
        for t,tt in enumerate(self.ds.rftime):
            if t<1: continue
            F = self.basalmelt(TMP[:t-1,:],gamma) #Units m/yr
            #dFdt = F[1:t,:]-F[:t-1,:]
            CRF = self.ds.irf[1:t,:].values #Units Gt/yr / m/yr
            #dCRF = (CRF[1:,:]-CRF[:-1,:]) #Units Gt/yr2 / m/yr
            #print(CRF.shape)
            #print(F.shape)
            #print(dCRF.shape)
            IML[t,:] = np.sum(CRF[::-1]*F,axis=0) #Units Gt
        
        #Calibration on cumulative IML
        err = (np.sum(IML[2017-self.year0,:])-np.sum(IML[1979-self.year0,:]))/4760
        
        #Output derivative: IML rate
        IML = np.diff(IML,axis=0,prepend=0) #Units Gt/yr
        return IML,err

    def oce2slr(self,TMP,gamma):
        #Conversion of ocean temperature anomaly to cumulative sea level rise
        
        #Define local variables
        assert TMP.shape == self.ds.srf.shape
        SLR = 0.*TMP
        
        #Integrate over time
        for t,tt in enumerate(self.ds.rftime):
            if t<1: continue
            F = self.basalmelt(TMP[:t-1,:],gamma) #Units m/yr
            CRF = self.ds.srf[1:t,:].values #Units m/yr / m/yr
            SLR[t,:] = np.sum(CRF[::-1]*F,axis=0) #Units m
            
        #Calibrate on total Antarctic period 1979-2017:
        err = np.sum(SLR[2017-self.year0,:]-SLR[1979-self.year0,:])/self.SLRtarget
        #calibrate on Amundsen:
        #err = (SLR[-1,2]-SLR[-41,2])/.0096
        return SLR,err
    
    def ice2oce(self,IML):
        #Conversion of ice mass loss rate to temperature anomaly
        
        #Define local variables
        assert IML.shape == self.ds.tanom.shape[:2]
        assert self.ds.tanom.shape[1] == self.ds.tanom.shape[2]
        dTMP = 0.*IML
        
        #Integrate over time
        assert self.nonlin in ['alpha','cutoff']
        if self.nonlin == 'alpha':
            alpha = [.55,.2,.4,.45,.45]
        elif self.nonlin == 'cutoff':
            alpha = [1,1,1,1,1]     
        
        Fmax = [840,520,560,740,770]
        
        for t,tt in enumerate(self.ds.rftime):
            if t<1: continue
            for e,ex in enumerate(self.ds.exp):
                #Index e now refers to source region of ice mass loss
                for b,bas in enumerate(self.ds.basin):
                    #IML in Gt/yr
                    F = IML[:t-1,e]
                    
                    #Scale for non-linear RF
                    F = np.sign(F)*np.abs(F)**alpha[b]
                    
                    if self.nonlin == 'cutoff':
                        F = np.maximum(np.minimum(F,Fmax[b]),-Fmax[b])
                    
                    #Compute ORF is time derivative of tanom / fanom in K/yr / (Gt/yr)^a
                    if self.usefanom: 
                        #Use exponential fit through tanom
                        CRF = (self.ds.fanom[1:t,e,b].values-self.ds.fanom[:t-1,e,b].values)/(self.pert**alpha[b])
                    else: 
                        #Use raw tanom
                        CRF = (self.ds.tanom[1:t,e,b].values-self.ds.tanom[:t-1,e,b].values)/(self.pert**alpha[b])
                            
                    dTMP[t,b] += np.sum(CRF[::-1]*F) #Sum over exp (= source region of ice mass loss)
            #print(t,TMP[t,:])
        #self.TMP[n,:,:] = self.TMP[0,:,:] + TMP#Add anomaly to temperature without feedback
        return dTMP
    
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
        
        self.year0 = ad.year0
        ds = ds.sel(time=slice(self.year0,self.year0+150-1))
        
        self.basin = ds.basin
        self.exp = ds.exp
        self.ism = ds.ism
        self.esm = ds.esm
        self.ssp = ds.ssp
        self.time = ds.time
        
        self.cal = 'S'
        
        self.nonlin = 'cutoff'
        
        return
    
    def gather(self,force_update=False):
        self.savename = f'../data/ensemble_cal{self.cal}_{self.nonlin}_{self.year0}.nc'
        if force_update:
            print('Doing a forced update of ensemble calculation: ',self.savename)
            self.compute()
        else:
            try:
                ds = xr.open_dataset(self.savename)
                print('Reading old saved ensemble ',self.savename)
                self.slr = ds['slr']
                self.gamma = ds['gamma']
                ds.close()
            except:
                print('New input parameters, calculating new ensemble: ',self.savename)
                self.compute()
        return
    
    def compute(self):
        c = 0 #Counter to show progress
        self.slr = np.zeros((2,len(self.ssp),len(self.esm),len(self.ism),len(self.time)))
        self.gamma = np.zeros((2,len(self.esm),len(self.ism)))
        
        for e,esm in enumerate(self.esm.values):
            for i,ism in enumerate(self.ism.values):
                #Calibrate to get gamma
                cens = EnsembleMember(self.ad,ism=ism,esm=esm,ssp='245',year0=1871,cal=self.cal,nonlin=self.nonlin)
                cens.calibrate_nf()
                cens.calibrate_wf()
                self.gamma[:,e,i] = cens.gamma
                for s,ssp in enumerate(self.ssp.values):
                    #Print progress
                    c+=1 
                    print(f'Computing esm {e+1} of {len(self.esm)} | ism {i+1} of {len(self.ism)} | ssp {s+1} of {len(self.ssp)} | Progress: {100*c/(len(self.ssp)*len(self.esm)*len(self.ism)):.0f}% ',end='           \r')
                    
                    #Calculation
                    ens = EnsembleMember(self.ad,ism=ism,esm=esm,ssp=ssp,year0=self.year0,cal=self.cal,nonlin=self.nonlin)
                    ens.compute(gamma=self.gamma[:,e,i])
                    self.slr[:,s,e,i,:] = np.sum(ens.SLR[:,:,:],axis=2)
                print(esm,ism,self.gamma[:,e,i],self.slr[1,:,e,i,-1]/self.slr[0,:,e,i,-1],end='                           \n')
                print(f'Computing esm {e+1} of {len(self.esm)} | ism {i+1} of {len(self.ism)} | ssp {s+1} of {len(self.ssp)} | Progress: {100*c/(len(self.ssp)*len(self.esm)*len(self.ism)):.0f}% ',end='           \r')
        self.save()
        return
    
    def save(self):  
        option = ['nf','wf']
        self.slr = xr.DataArray(self.slr,dims=('option','ssp','esm','ism','time'),coords={'option':option,'ssp':self.ssp,'esm':self.esm,'ism':self.ism,'time':self.time},attrs={'unit':'m','long_name':'Cumulative sea level rise'})
        self.gamma = xr.DataArray(self.gamma,dims=('option','esm','ism'),coords={'option':option,'esm':self.esm,'ism':self.ism},attrs={'unit':'m/yr/K^2','long_name':'Basal melt parameter'})
        
        self.ds = xr.Dataset({'slr':self.slr,'gamma':self.gamma})
        self.ds.to_netcdf(self.savename,mode='w')
        self.ds.close()
        print('Saved',self.savename,'                                           ')
        
        
        
        
        
        
class FullEnsemble2(Constants):
    """ 
    Input: ad: AllData class
    
    """
    def __init__(self,ad):
        #Read constants
        Constants.__init__(self)
        
        #Read input
        self.ad = ad
        ds = ad.ds.isel(exp=slice(0,5),basin=slice(0,5))
        
        self.year0 = ad.year0
        ds = ds.sel(time=slice(self.year0,self.year0+150-1))
        
        self.basin = ds.basin
        self.exp = ds.exp
        self.ism = ds.ism
        self.esm = ds.esm
        self.ssp = ds.ssp
        self.time = ds.time
        
        self.gamma = 2.57 #Fixed value
        
        self.nonlin = 'cutoff'
        
        return
    
    def gather(self,force_update=False):
        
        self.savename = f'../data/ensemble_fixed_{self.nonlin}_{self.year0}.nc'
        
        if force_update:
            print('Doing a forced update of ensemble calculation: ',self.savename)
            self.compute()
        else:
            try:
                ds = xr.open_dataset(self.savename)
                print('Reading old saved ensemble ',self.savename)
                self.slr = ds['slr']
                ds.close()
            except:
                print('New input parameters, calculating new ensemble: ',self.savename)
                self.compute()
        return
    
    def compute(self):
        c = 0 #Counter to show progress
        self.slr = np.zeros((2,len(self.ssp),len(self.esm),len(self.ism),len(self.time)))
                
        for e,esm in enumerate(self.esm.values):
            for i,ism in enumerate(self.ism.values):
                for s,ssp in enumerate(self.ssp.values):
                    #Print progress
                    c+=1 
                    print(f'Computing esm {e+1} of {len(self.esm)} | ism {i+1} of {len(self.ism)} | ssp {s+1} of {len(self.ssp)} | Progress: {100*c/(len(self.ssp)*len(self.esm)*len(self.ism)):.0f}% ',end='           \r')

                    #Calculation
                    ens = EnsembleMember(self.ad,ism=ism,esm=esm,ssp=ssp,year0=self.year0,nonlin=self.nonlin)
                    ens.compute(gamma=np.array([self.gamma,self.gamma]),recal=False)
                    self.slr[:,s,e,i,:] = np.sum(ens.SLR[:2,:,:],axis=2)
                print(esm,ism,self.slr[1,:,e,i,-1]/self.slr[0,:,e,i,-1],end='                                          \n')
                print(f'Computing esm {e+1} of {len(self.esm)} | ism {i+1} of {len(self.ism)} | ssp {s+1} of {len(self.ssp)} | Progress: {100*c/(len(self.ssp)*len(self.esm)*len(self.ism)):.0f}% ',end='           \r')
        self.save()
        return
    
    def save(self):  
        opt2 = ['nf','wf']
        self.slr = xr.DataArray(self.slr,dims=('opt2','ssp','esm','ism','time'),coords={'opt2':opt2,'ssp':self.ssp,'esm':self.esm,'ism':self.ism,'time':self.time},attrs={'unit':'m','long_name':'Cumulative sea level rise'})
        self.ds = xr.Dataset({'slr':self.slr})
        self.ds.to_netcdf(self.savename,mode='w')
        self.ds.close()
        print('Saved',self.savename,'                                           ')