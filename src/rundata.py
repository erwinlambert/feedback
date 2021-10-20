import numpy as np
import xarray as xr
import glob

from constants import Constants

class RunData(Constants):
    """ Data and attributes for a single run 
    
    """
    def __init__(self,run):
        #Read input
        self.run = run

        #Read constants
        Constants.__init__(self)
        
        #Other stuff
        self.savename_mon = f'../data/temperature_mon_{self.run}.nc'
        self.savename_ann = f'../data/temperature_ann_{self.run}.nc'
        
        return
    
    def checkfornewdata(self,verbose=True,option='all'):
        assert option in ['all','ann','mon']
        if verbose:
            print(f'Checking for {option} data from {self.run}')
        self.get_fnames()
        if option in ['all','ann']:
            self.get_nyears()
            if self.nyears_saved<self.nyears:
                if verbose:
                    print(f'Updating annual time series with {self.nyears-self.nyears_saved} years ...')
                self.update_annual()
            if verbose:
                print('Annual data complete')
        
        if option in ['all','mon']:
            self.get_nmonths()    
            if self.nmonths_saved<self.nmonths:
                if verbose:
                    print(f'Updating monthly time series with {self.nmonths-self.nmonths_saved} months ...')
                self.update_monthly()
            if verbose:
                print('Monthly data complete')
        if verbose:
            print('---------------------')
        return
    
    def get_nyears(self):
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
        #print('Annual data: saved',self.nyears_saved,'of',self.nyears,'years')
        return 
    
    def get_nmonths(self):
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
        #print('Monthly data: saved',self.nmonths_saved,'of',self.nmonths,'months')
        return
    
    def get_fnames(self):
        if self.run == 'ctrl':
            self.fnames = sorted(glob.glob(f'../data/ecefiles/n011/n011*T.nc'))[38:-5]
        elif self.run == 'spin':
            self.fnames = sorted(glob.glob(f'../data/ecefiles/n011/n011*T.nc'))[:38]
        else:
            self.fnames = sorted(glob.glob(f'../data/ecefiles/{self.run}/{self.run}*T.nc'))
        return 

    def update_monthly(self):
    
        ds = xr.open_dataset('../data/basinvolumes.nc')
        volume = ds['volume']
        ds.close()

        #Calculate basin-average annual time series
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
        
        c = 0
        for f,fname in enumerate(self.fnames):
            ds = xr.open_dataset(fname)
            time = ds['time_centered'].values
            if c+len(time)<self.nmonths_saved:
                ds.close()
                c+=len(time)
                print(f'Skipping {c:3.0f}',end='                     \r')
                continue
            else:
                temp = ds['thetao'].values
                ds.close()

            for m in range(len(time)):
                if c<self.nmonths_saved:
                    c+=1
                    print(f'Skipping {c:3.0f}',end='                     \r')
                    continue
                for b,bas in enumerate(volume.basin):
                    tbas[c,b] = np.nansum(temp[m,:,:,:]*volume.sel(basin=bas))/np.nansum(volume.sel(basin=bas))
                print(f'{c:3.0f}',tbas[c,:],end='                                                \r')
                c += 1

        #Save data
        temp2 = xr.DataArray(tbas,dims=('time','basin'),coords={'time':ttime,'basin':volume.basin},attrs={'unit':'degrees Celcius','long_name':'temperature time series per basin'})
        ds = xr.Dataset({'temp':temp2})
        ds.to_netcdf(self.savename_mon,mode='w')
        ds.close()
        print('Updated monthly time series in',self.savename_mon,end='                     \n')
        return
    
    def update_annual(self):
    
        ds = xr.open_dataset('../data/basinvolumes.nc')
        volume = ds['volume']
        ds.close()

        #Calculate basin-average annual time series
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
        
        c = 0
        for f,fname in enumerate(self.fnames):
            ds = xr.open_dataset(fname)
            time = ds['time_centered'].values
            ny = int(len(time)/12)
            
            if c+ny<self.nyears_saved:
                ds.close()
                c+=ny
                print(f'Skipping {c:3.0f}',end='                     \r')
                continue
            else:
                temp = ds['thetao'].values
                ds.close()

            year0 = int(fname[-27:-23])

            tb = np.zeros((len(self.basin)))
            
            for y in np.arange(0,ny):
                if c<self.nyears_saved:
                    c+=1
                    print(f'Skipping {c:3.0f}',end='                     \r')
                    continue                

                for b,bas in enumerate(self.basin):
                    for m,mm in enumerate(self.months):
                        tbb = np.nansum(temp[m+12*y,:,:,:]*volume.sel(basin=bas))/np.nansum(volume.sel(basin=bas))
                        tb[b] += tbb*self.spm[m]
                    tbas[c,b] = tb[b]/sum(self.spm)
                print(year0+y,f'{c:3.0f}',tbas[c,:],end='                                \r')
                tb = np.zeros((len(self.basin)))
                c += 1

        temp2 = xr.DataArray(tbas,dims=('time','basin'),coords={'time':ttime,'basin':self.basin},attrs={'unit':'degrees Celcius','long_name':'temperature time series per basin'})
        ds = xr.Dataset({'temp':temp2})
        ds.to_netcdf(self.savename_ann,mode='w')
        ds.close()
        print('Updated annual time series in',self.savename_ann) 
        return