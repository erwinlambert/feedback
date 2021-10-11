import xarray as xr
import numpy as np

import utils as ut

def save_basinvolumes():
    #Get time-independent variables
    ds = xr.open_dataset('../data/ecefiles/areas.nc')
    area = ds['O1t0.srf'].values;
    ds.close()
    
    ds = xr.open_dataset('../data/ecefiles/n011/n011_1m_18500101_18501231_grid_T.nc')
    y = ds['y'].values
    x = ds['x'].values
    lat = ds['nav_lat'].values
    lon = ds['nav_lon'].values
    levmid = ds['olevel'].values
    lev = ds['olevel_bounds'].values
    time_bnds = ds['time_centered_bounds']
    thick = ds['e3t'].isel(time_counter=0).values #Quasi-time-independent, treated as fixed
    ds.close()
    secs = (time_bnds[:12,1]-time_bnds[:12,0]).values / np.timedelta64(1, 's')
    
    #To check relative volume:
    basins = np.append(ut.basin,['Pens1','Pens2'])

    #Get mask for basin averages
    lons = np.repeat(lon[np.newaxis,:,:],len(levmid),axis=0)
    lats = np.repeat(lat[np.newaxis,:,:],len(levmid),axis=0)
    volume = np.repeat(lats[np.newaxis,:,:,:],len(basins),axis=0)

    for b,bas in enumerate(basins):
        mm = np.zeros(lons.shape)
        if bas=='East Ant.':
            #EAIS
            mm[((lons<173) & (lons>-10)) & (lats<-65) & (lats>-76)] = 1
            depp = 369
        if bas=='Ross':
            #ROSS
            mm[((lons>150) | (lons<-150)) & (lats<-76)] = 1
            depp = 312        
        if bas=='Amundsen':
            #AMUN
            mm[(lons>-150) & (lons<-80) & (lats<-70)] = 1
            depp = 305
        if bas=='Weddell':
            #WEDD
            mm[(lons>-65) & (lons<-10) & (lats<-72)] = 1
            depp = 420
        if bas=='Peninsula':
            #APEN
            mm[(lons>-66) & (lons<-56) & (lats>-70) & (lats<-65)] = 1
            mm[(lons>-80) & (lons<-65) & (lats>-75) & (lats<-70)] = 1
            depp = 420
        if bas=='Pens1':
            #Eastern peninsula
            mm[(lons>-66) & (lons<-56) & (lats>-70) & (lats<-65)] = 1
            depp = 420
        if bas=='Pens2':
            #Western peninsula
            mm[(lons>-80) & (lons<-65) & (lats>-75) & (lats<-70)] = 1
            depp = 420

        z0 = depp-50.
        i0 = np.argmax(lev[:,1]>z0)
        mm[:i0,:,:] = 0
        w0 = (lev[i0,1]-z0)/(lev[i0,1]-lev[i0,0])
        mm[i0,:,:] = w0*mm[i0,:,:]
        for j in range(0,lon.shape[0]):
            for i in range(0,lon.shape[1]):
                if np.nansum(thick[i0:,j,i]) == 0:
                    continue
                z1 = depp+50.
                i1 = np.argmin(lev[:,1]<z1)
                w1 = (z1-lev[i1,0])/(lev[i1,1]-lev[i1,0])
                mm[i1,j,i] = w1*mm[i1,j,i]
                mm[i1+1:,j,i] = 0
        volume[b,:,:,:] = mm*np.where(np.isnan(thick),0,thick)*area[np.newaxis,:,:]
    
    volume2 = xr.DataArray(volume,dims=('basin','lev','y','x'),attrs={'unit':'m^3','long_name':'Gridded 3D volume per basin'})
    ds = xr.Dataset({'volume':volume2})
    ds = ds.assign_coords({"basin":(['basin'],basins),"lev":(['lev'],levmid),"lat": (["y", "x"], lat),"lon": (["y", "x"], lon)})
    ds.to_netcdf('../data/basinvolumes.nc',mode='w')
    ds.close()
    print('Saved basin volumes')