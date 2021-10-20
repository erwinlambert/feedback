import numpy as np
import xarray as xr

from constants import Constants
from rundata import RunData

class AllData(Constants):
    """ Combined time series from all runs, CMIP6 temperatures, and IRFs and SRFs
    
    """
    def __init__(self,verbose=False):
        
        #Read constants
        Constants.__init__(self)
        
        #New constants
        self.year0 = 1871
        self.nyears = 150
        self.option = '' #Whether to determine ORFs wrt linear fit through control run
        
        self.basin = self.basin[:5] #Omit separate East and West Peninsula
        
        #Ice sheet models
        self.ism   = ['CISM_NCA','FETI_ULB','GRIS_LSC','IMAU_UU','ISSM_UCI','MALI_DOE','PISM_DMI','PISM_PIK','SICO_ILTS']
        self.ism_i = ['CISM_NCA','FETISH_ULB','GRIS_LSC','IMAU_VUB','ISSM_UCI','MALI_DOE','PISM_DMI','PISM_PIK','SICO_ILTS']
        self.ism_s = ['CISM_NCA','FETI_VUB','GRIS_LSC','IMAU_VUB','ISSM_UCI','MALI_LAN','PISM_DMI','PISM_PIK','SICO_UHO']
        self.region = ['R1','R2','R3','R4','R5']
        
        #Option for historical data
        self.histop = 'detrend_mix_biasadj_'
        self.evbas = ['eais','ross','amun','wedd','apen']
        
        #Parameters for dummy ORFs
        self.mag = [.7,1,.3,.6,.8] #degC
        self.tsc = [30,50,100,60,50] #year
        self.faclo = .1 #Factor of remote response
        self.fachi = .6 #Factor of local response
        
        self.verbose = verbose
        
        return
    
    def gather(self,update=True):
        self.year1 = self.year0+self.nyears-1
        self.rftime = np.arange(self.nyears)
        
        if update: self.update_runs()
        
        self.get_external()
        self.get_tanom()
        
        irf2  = xr.DataArray(self.irf,dims=('rftime','ism','basin'),coords={'rftime':self.rftime,'ism':self.ism,'basin':self.basin},attrs={'unit':'Gt/yr per m/yr','long_name':'ice mass loss response function to increased basal melt'})
        srf2  = xr.DataArray(self.srf,dims=('rftime','ism','basin'),coords={'rftime':self.rftime,'ism':self.ism,'basin':self.basin},attrs={'unit':'m/yr per m/yr','long_name':'sea level response function to increased basal melt'})
        temp2 = xr.DataArray(self.temp,dims=('time','esm','ssp','basin'),coords={'time':self.time,'esm':self.esm,'ssp':self.ssp,'basin':self.basin},attrs={'unit':'degrees Celcius','long_name':'temperature anomaly from pre-industrial control'})
        tanom2  = xr.DataArray(self.tanom,dims=('rftime','exp','basin'),coords={'rftime':self.rftime,'exp':self.exp,'basin':self.basin},attrs={'unit':'degrees Celcius','long_name':'ocean response function to increased mass loss'})

        ds = xr.Dataset({'irf':irf2,'srf':srf2,'temp':temp2,'tanom':tanom2})
        
        if self.verbose:
            print('-----------------------------------------------')
        print(f'Combined all data from {self.year0} to {self.year1} with option {self.option}')
        return ds
    
    def get_external(self):
        #Get external data
        self.get_irf()
        self.get_srf()
        self.get_cmip6temp()
        return
    
    def get_tanom(self):
        self.get_tanom_dummy() #Temperature anomalies from hypothetical data
        self.overwrite_tanom_fromtotl() #Temperature anomalies derived from TOTL run, hypothetical distribution
        self.overwrite_tanom_real() #Temperature anomalies from actual output
        return
    
    def update_runs(self,option='ann'):
        for ee in self.exp:
            rd = RunData(ee)
            rd.checkfornewdata(option=option,verbose=self.verbose)
        return
    
    def get_irf(self):
        #Read IRF
        self.irf  = np.zeros((len(self.rftime),len(self.ism),len(self.basin)))
        for i,ii in enumerate(self.ism_i):
            for r,reg in enumerate(self.region):
                with open(f'../data/Larmip/RFunctions_total/RF_{ii}_BM08_{reg}.dat') as f:
                    self.irf[:,i,r] = np.array([float(x) for x in f.readlines()])[:self.nyears]
        return

    def get_srf(self):
        #Read SRF
        self.srf  = np.zeros((len(self.rftime),len(self.ism),len(self.basin)))
        for i,ii in enumerate(self.ism_s):
            for r,reg in enumerate(self.region):
                with open(f'../data/Larmip/RFunctions/RF_{ii}_BM08_{reg}.dat') as f:
                    self.srf[:,i,r] = np.array([float(x) for x in f.readlines()])[:self.nyears]
        return
    
    def get_cmip6temp(self): 
        #Get available models and time range
        mods = {}
        for s,ss in enumerate(self.ssp):
            ds = xr.open_dataset(f'../data/eveline/thetao_{self.histop}sector_timeseries_historical+ssp{ss}_1850_2100.nc')
            mods[ss] = ds.model.values
            if s==0:
                #Get time coordinates
                ds = ds.sel(year=slice(self.year0,self.year1))
                self.time = ds.year.values
            ds.close()
        allmods = set(mods[self.ssp[0]]) & set(mods[self.ssp[1]]) & set(mods[self.ssp[2]])
        self.esm = sorted(np.array([mod for mod in allmods]))
        
        #Read temperature
        self.temp = np.zeros((len(self.time),len(self.esm),len(self.ssp),len(self.basin)))
        for s,ss in enumerate(self.ssp):
            ds = xr.open_dataset(f'../data/eveline/thetao_{self.histop}sector_timeseries_historical+ssp{ss}_1850_2100.nc')
            ds = ds.sel(year=slice(self.year0,self.year1))
            for e,es in enumerate(self.esm):
                for b,eb in enumerate(self.evbas):
                    self.temp[:,e,s,b] = ds[eb].sel(model=es).values
                #print(ss,es,temp[-1,e,s,:])
            ds.close()
        return
    
    def get_tanom_dummy(self):
        self.tanom  = np.zeros((len(self.rftime),len(self.exp),len(self.basin)))

        #Magnitude and time scale of response 

        for b,bas in enumerate(self.basin):
            for e,ee in enumerate(self.exp[:5]):
                if b==e:
                    fac = self.fachi
                else:
                    fac = self.faclo
                self.tanom[:,e,b] = fac* self.mag[b]*(1-np.exp(-self.rftime/self.tsc[b]))

        for b,bas in enumerate(self.basin):
            self.tanom[:,5,b] = np.sum(self.tanom[:,:5,b],axis=1)

        self.tanom[:,6,:] = .5* self.tanom[:,5,:]
        self.tanom[:,7,:] =  2* self.tanom[:,5,:]
        if self.verbose:
            print('Created dummy data for temperature anomalies')
        return
    
    def overwrite_tanom_fromtotl(self):
        
        ds = xr.open_dataset(f'../data/temperature_ann_ctrl.nc').isel(basin=slice(0,5))
        tctrl = ds.temp
        ds.close()

        #Overwrite control with linear fit
        if self.option=='_fit':
            for b,bas in enumerate(self.basin):
                out = np.polyfit(self.rftime,tctrl[:,b],1)
                tctrl[:,b] = out[1] + self.rftime*out[0]

        ds = xr.open_dataset(f'../data/temperature_ann_tot2.nc').isel(basin=slice(0,5))
        yavail = len(ds.time)
        dtemp = ds.temp - tctrl.isel(time=slice(0,yavail))
        ds.close()

        #Individual basin responses
        for b,bas in enumerate(self.basin):
            for e,ee in enumerate(self.exp[:5]):
                if b==e:
                    fac = self.fachi
                else:
                    fac = self.faclo
                self.tanom[:yavail,e,b] = fac* dtemp.sel(basin=bas)

        #TOTL
        self.tanom[:yavail,5,:] = dtemp

        #HALF and DOUB
        self.tanom[:yavail,6,:] = .5* self.tanom[:yavail,5,:]
        self.tanom[:yavail,7,:] =  2* self.tanom[:yavail,5,:]
        if self.verbose:
            print(f'Overwrote first {yavail} years with TOTL-derived estimates')
        return
    
    def overwrite_tanom_real(self):
        #Overwrite temperature anomalies from actual output
        for e,ee in enumerate(self.exp):
            try:
                ds = xr.open_dataset(f'../data/temperature_ann_{ee}.nc')
                yavail = len(ds.time)
                if self.verbose:
                    print(f'Overwrote first {yavail} years with real data from run {ee}')
                self.tanom[:yavail,e,:] = ds.temp - tctrl.isel(time=slice(0,yavail))
                ds.close()
            except:
                pass
        return