import numpy as np
import xarray as xr
from scipy.optimize import curve_fit


from constants import Constants
from rundata import RunData

class AllData(Constants):
    """ Combined time series from all runs, CMIP6 temperatures, and IRFs and SRFs
    
    """
    def __init__(self,verbose=False):
        
        #Read constants
        Constants.__init__(self)
        
        #New constants
        self.nyears = 150
        self.fitctrl = True
        
        self.basin = self.basin[:5] #Omit separate East and West Peninsula
        
        #Ice sheet models
        self.ism   = ['CISM_NCA','FETI_ULB','GRIS_LSC','IMAU_UU','ISSM_UCI','MALI_DOE','PISM_DMI','PISM_PIK','SICO_ILTS']
        self.ism_i = ['CISM_NCA','FETISH_ULB','GRIS_LSC','IMAU_VUB','ISSM_UCI','MALI_DOE','PISM_DMI','PISM_PIK','SICO_ILTS']
        self.ism_s = ['CISM_NCA','FETI_VUB','GRIS_LSC','IMAU_VUB','ISSM_UCI','MALI_LAN','PISM_DMI','PISM_PIK','SICO_UHO']
        self.region = ['R1','R2','R3','R4','R5']
        
        #Option for historical data
        self.histop = 'detrend_mix_biasadj_'
        self.evbas = ['eais','ross','amun','wedd','apen']
        
        self.verbose = verbose
        
        return
    
    def gather(self,update=True):
        self.rftime = np.arange(self.nyears)
        
        if update: self.update_runs()
        
        self.get_external()
        self.get_tanom()
        self.get_fanom()
        
        irf2  = xr.DataArray(self.irf,dims=('rftime','ism','basin'),coords={'rftime':self.rftime,'ism':self.ism,'basin':self.basin},attrs={'unit':'Gt/yr per m/yr','long_name':'ice mass loss response function to increased basal melt'})
        srf2  = xr.DataArray(self.srf,dims=('rftime','ism','basin'),coords={'rftime':self.rftime,'ism':self.ism,'basin':self.basin},attrs={'unit':'m/yr per m/yr','long_name':'sea level response function to increased basal melt'})
        temp2 = xr.DataArray(self.temp,dims=('time','esm','ssp','basin'),coords={'time':self.time,'esm':self.esm,'ssp':self.ssp,'basin':self.basin},attrs={'unit':'degrees Celcius','long_name':'temperature anomaly from pre-industrial control'})
        tanom2  = xr.DataArray(self.tanom,dims=('rftime','exp','basin'),coords={'rftime':self.rftime,'exp':self.exp,'basin':self.basin},attrs={'unit':'degrees Celcius','long_name':'ocean response function to increased mass loss'})
        fanom2  = xr.DataArray(self.fanom,dims=('rftime','exp','basin'),coords={'rftime':self.rftime,'exp':self.exp,'basin':self.basin},attrs={'unit':'degrees Celcius','long_name':'fitted ocean response function to increased mass loss'})
        tref2  = xr.DataArray(self.tref,dims=('esm','basin'),coords={'esm':self.esm,'basin':self.basin},attrs={'unit':'degrees Celcius','long_name':'historical reference temperature per ESM per basin'})
        
        self.ds = xr.Dataset({'irf':irf2,'srf':srf2,'temp':temp2,'tanom':tanom2,'fanom':fanom2,'tref':tref2})
        
        if self.verbose:
            print('-----------------------------------------------')
        print(f'Gathered all data')
        return
    
    def get_external(self):
        #Get external data
        self.get_irf()
        self.get_srf()
        self.get_cmip6temp()
        return
    
    def get_tanom(self):
        #Initialise tanom variable
        self.tanom  = np.zeros((len(self.rftime),len(self.exp),len(self.basin)))

        #Get control time series
        ds = xr.open_dataset(f'../data/temperature_ann_ctrl.nc').isel(basin=slice(0,5))
        tctrl = ds.temp
        ds.close()

        #Overwrite control with linear fit
        if self.fitctrl:
            for b,bas in enumerate(self.basin):
                out = np.polyfit(self.rftime,tctrl[:,b],1)
                tctrl[:,b] = out[1] + self.rftime*out[0]
        
        #Overwrite temperature anomalies from actual output
        for e,ee in enumerate(self.exp):
            ds = xr.open_dataset(f'../data/temperature_ann_{ee}.nc')
            yavail = len(ds.time)
            if self.verbose:
                print(f'Diagnosed first {yavail} years of real data from run {ee}')
            self.tanom[:yavail,e,:] = ds.temp - tctrl.isel(time=slice(0,yavail))
            ds.close()
        
    def get_fanom(self):
        #Exponential fit through tanom
        self.fanom  = np.zeros((len(self.rftime),len(self.exp),len(self.basin)))

        f = lambda t, *p: p[0] * (1-np.exp(-t/p[1]))
        
        for e,ee in enumerate(self.exp):
            ds = xr.open_dataset(f'../data/temperature_ann_{ee}.nc')
            yavail = len(ds.time)
            ds.close()
            for b,bas in enumerate(self.basin):
                popt, pcov = curve_fit(f,self.rftime[:yavail],self.tanom[:yavail,e,b],[1,50])
                self.fanom[:,e,b] = popt[0]*(1-np.exp(-self.rftime/popt[1]))
            if self.verbose:
                print(f'Created exponential fit through first {yavail} years of tanom for run {ee}')
        return
    
    def update_runs(self,option='ann'):
        for ee in self.exp:
            rd = RunData(ee,verbose=self.verbose)
            rd.checkfornewdata(option=option)
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
                self.time = ds.year.values
            ds.close()
        allmods = set(mods[self.ssp[0]]) & set(mods[self.ssp[1]]) & set(mods[self.ssp[2]])
        self.esm = sorted(np.array([mod for mod in allmods]))
        
        #Read temperature
        self.temp = np.zeros((len(self.time),len(self.esm),len(self.ssp),len(self.basin)))
        self.tref = np.zeros((len(self.esm),len(self.basin)))
        for s,ss in enumerate(self.ssp):
            ds = xr.open_dataset(f'../data/eveline/thetao_{self.histop}sector_timeseries_historical+ssp{ss}_1850_2100.nc')
            if s==0:
                for e,es in enumerate(self.esm):
                    for b,eb in enumerate(self.evbas):
                        self.tref[e,b] = np.mean(ds[eb].sel(model=es).values[:100])            
            for e,es in enumerate(self.esm):
                for b,eb in enumerate(self.evbas):
                    self.temp[:,e,s,b] = ds[eb].sel(model=es).values
            ds.close()
        return