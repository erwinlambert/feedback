import numpy as np
import xarray as xr
import glob

from constants import Constants

class RunData(Constants):
    """ 
    Contains metadata of output from a single run
    
    Allows for analysis of new available raw data
    
    Allows for writing out basin temperature time series (annual and monthly)
    
    """
    def __init__(self,run,verbose=True):
        #Read input
        self.run = run

        #Read constants
        Constants.__init__(self)
        
        #Other stuff
        self.savename_mon = f'../data/temperature_mon_{self.run}.nc'
        self.savename_ann = f'../data/temperature_ann_{self.run}.nc'
        
        #Determine whether to print a lot of updates
        self.verbose=verbose
        
        #Get some metadata
        self.get_fnames()
        self.get_nmonths()
        self.get_nyears()
        
        self.jmax = 55 #Maximum latitude index for 2d variables
        self.var2d = {}
        self.dep2d = {}
        
        return
    
    def checkfornewdata(self,option='all'):
        #Check for new annual and/or monthly data from runs from which no basin temperatures are saved yet
        assert option in ['all','ann','mon']
        if self.verbose: print(f'Checking for {option} data from {self.run}')
        #Get file names of raw data
        self.get_fnames()
        
        if option in ['all','ann']:
            #Update annual basin temperatures
            self.get_nyears()
            if self.nyears_saved<self.nyears: #Check whether any new data is available
                print(f'Updating annual time series with {self.nyears-self.nyears_saved} years ...')
                self.update_annual()
            if self.verbose: print('Annual data complete')
        
        if option in ['all','mon']:
            #Update monthly basin temperatures
            self.get_nmonths()    
            if self.nmonths_saved<self.nmonths: #Check whether any new data is available
                print(f'Updating monthly time series with {self.nmonths-self.nmonths_saved} months ...')
                self.update_monthly()
            if self.verbose: print('Monthly data complete')
        if self.verbose: print('---------------------')
        return
    
    def get_nyears(self):
        #Compute the number of available years
        self.nyears = 0
        for fname in self.fnames:
            ds = xr.open_dataset(fname)
            self.nyears += int(np.floor(len(ds.time_counter)/12))
            ds.close()
        try:
            ds = xr.open_dataset(self.savename_ann)
            self.nyears_saved = len(ds.time)
            ds.close()
        except:
            self.nyears_saved = 0
        return 
    
    def get_nmonths(self):
        #Compute the number of available months
        self.nmonths = 0
        for fname in self.fnames:
            ds = xr.open_dataset(fname)
            self.nmonths += len(ds.time_counter)
            ds.close()
        try:
            ds = xr.open_dataset(self.savename_mon)
            self.nmonths_saved = len(ds.time)
            ds.close()
        except:
            self.nmonths_saved = 0
        return
    
    def get_fnames(self):
        #Get the filenames containing the raw data
        if self.run == 'ctrl':
            self.fnames = sorted(glob.glob(f'../data/ecefiles/n011/n011*T.nc'))[38:-5]
        elif self.run == 'spin':
            self.fnames = sorted(glob.glob(f'../data/ecefiles/n011/n011*T.nc'))[:38]
        else:
            self.fnames = sorted(glob.glob(f'../data/ecefiles/{self.run}/{self.run}*T.nc'))
        return 

    def update_monthly(self):
        #Update the monthly time series of basin temperatures
        
        #Read basin volumes for spatial averaging
        ds = xr.open_dataset('../data/basinvolumes.nc')
        volume = ds['volume']
        ds.close()

        #Allocate variables
        tbas = np.zeros((self.nmonths,len(volume.basin)))
        ttime = np.arange(self.nmonths)

        #Get already saved data
        try:
            ds = xr.open_dataset(self.savename_mon)
            tbas[:self.nmonths_saved,:] = ds.temp
            ttime[:self.nmonths_saved] = ds.time
            ds.close()
        except:
            pass
        
        #Get new mothly data
        c = 0 #Month counter
        for f,fname in enumerate(self.fnames):
            ds = xr.open_dataset(fname)
            time = ds['time_centered'].values
            if c+len(time)<self.nmonths_saved:
                #All data from this file is already saved
                ds.close()
                c+=len(time)
                if self.verbose: print(f'Skipping {c:3.0f}',end='                     \r')
                continue
            else:
                temp = ds['thetao'].values
                ds.close()

            for m in range(len(time)):
                if c<self.nmonths_saved:
                    #Data for this month is already saved
                    c+=1
                    if self.verbose: print(f'Skipping {c:3.0f}',end='                     \r')
                    continue
                for b,bas in enumerate(volume.basin):
                    tbas[c,b] = np.nansum(temp[m,:,:,:]*volume.sel(basin=bas))/np.nansum(volume.sel(basin=bas))
                if self.verbose: print(f'{c:3.0f}',tbas[c,:],end='                                                \r')
                c += 1

        #Save data
        temp2 = xr.DataArray(tbas,dims=('time','basin'),coords={'time':ttime,'basin':volume.basin},attrs={'unit':'degrees Celcius','long_name':'temperature time series per basin'})
        ds = xr.Dataset({'temp':temp2})
        ds.to_netcdf(self.savename_mon,mode='w')
        ds.close()
        if self.verbose: print('Updated monthly time series in',self.savename_mon,end='                     \n')
        return
    
    def update_annual(self):
        #Update the annual time series of basin temperatures
        
        #Read basin volumes for spatial averaging
        ds = xr.open_dataset('../data/basinvolumes.nc')
        volume = ds['volume']
        ds.close()

        #Allocate variables
        tbas = np.zeros((self.nyears,len(self.basin)))
        ttime = np.arange(self.nyears)

        #Get already saved data
        try:
            ds = xr.open_dataset(self.savename_ann)
            tbas[:self.nyears_saved,:] = ds.temp
            ttime[:self.nyears_saved] = ds.time
            ds.close()
        except:
            pass
        
        #Get new mothly data
        c = 0 #Year counter
        for f,fname in enumerate(self.fnames):
            ds = xr.open_dataset(fname)
            time = ds['time_centered'].values
            ny = int(len(time)/12) #Number of years in this file
            
            if c+ny<self.nyears_saved:
                #All data from this file is already saved
                ds.close()
                c+=ny
                if self.verbose: print(f'Skipping {c:3.0f}',end='                     \r')
                continue
            else:
                temp = ds['thetao'].values
                ds.close()

            year0 = int(fname[-27:-23]) #First year in file from file name
            
            for y in np.arange(0,ny):
                if c<self.nyears_saved:
                    #Data for this year is already saved
                    c+=1
                    if self.verbose: print(f'Skipping {c:3.0f}',end='                     \r')
                    continue                

                #Allocate variable to accumulate monthly temperatures
                tb = np.zeros((len(self.basin)))
                    
                for b,bas in enumerate(self.basin):
                    for m,mm in enumerate(self.months):
                        #Get volume-weighted average temperature for this month
                        tbb = np.nansum(temp[m+12*y,:,:,:]*volume.sel(basin=bas))/np.nansum(volume.sel(basin=bas))
                        #Accumulate monthly value, weighted by number of seconds per month
                        tb[b] += tbb*self.spm[m]
                    #Divide accumulated values by number of seconds per month to get annual average
                    tbas[c,b] = tb[b]/sum(self.spm)
                if self.verbose: print(year0+y,f'{c:3.0f}',tbas[c,:],end='                                \r')
                c += 1

        #Save data
        temp2 = xr.DataArray(tbas,dims=('time','basin'),coords={'time':ttime,'basin':self.basin},attrs={'unit':'degrees Celcius','long_name':'temperature time series per basin'})
        ds = xr.Dataset({'temp':temp2})
        ds.to_netcdf(self.savename_ann,mode='w')
        ds.close()
        if self.verbose: print('Updated annual time series in',self.savename_ann) 
        return
    
    def get_var2d(self,vname,depth,maxmon=999999):
        nmon = min(maxmon,self.nmonths)
        #Allocate variable
        ds = xr.open_dataset(self.fnames[0]).isel(y=slice(0,self.jmax))
        olev = np.argmin((ds['olevel'].values-depth)**2)
        self.lon = ds.nav_lon
        self.lat = ds.nav_lat
        self.dep2d[vname] = ds['olevel'][olev].values
        self.var2d[vname] = np.zeros((nmon,self.lon.shape[0],self.lon.shape[1]))
        ds.close()
        print(f'Getting {nmon} months of {vname} at depth {self.dep2d[vname]:.0f}m')
        c = -1
        for fname in self.fnames:
            ds = xr.open_dataset(fname).isel(y=slice(0,self.jmax))
            for t,tt in enumerate(ds['time_counter'].values):
                c+=1
                if c>=nmon:
                    break
                self.var2d[vname][c,:,:] = ds[vname].isel(olevel=olev,time_counter=t)
                print(f'Got {vname}: month {c+1} of {nmon}',end='       \r')
            ds.close()
            if c>=nmon:
                break
        print(f'\rGot {nmon} months of {vname} at depth {self.dep2d[vname]:.0f}m')
        return