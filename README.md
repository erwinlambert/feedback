## feedback project

All code is run from notebooks in `notebooks/`

Some functions and input are in `src/utils` and `src/utils2`

All data is stored in `data/`

---

#### Preparation for EC-Earth runs:
> Make_masks: Create basal melt and calving masks 
>> Output: `data/inputfiles/runoff_maps.nc`
> Make_depthfile: Create mask for basal melt depth (200m)
>> Output: `data/inputfiles/runoff_depth.nc`
> In addition, source code for EC-Earth is in `src/runoff_mapper_mod_****`

#### Preparation for post-processing:
> Read_cmip6: Get basin temperatures (period: piControl or historical)
>> Output: `data/temperature_cmip6_{period}.nc`

#### Actual post-processing:
> Read_ece_ts_oceantemp: Get basin temperatures from new runs (freq: mon (monthly) or ann (annual))
>> Output: `data/temperature_{freq}_{run}.nc`
>> Also creates: `data/basinvolumes.nc` for weighted averaging in basins
> Read_alldata: Combine response functions and temperatures into one file (year0: 1850 or 1950)
>> Output: `data/alldata_{year0}_{year1}.nc`
> Calc_ensemble: Calculate sea-level ensemble projections (year0: 1850 or 1950, bmp: lin or quad)
> This creates sea level trends with and without feedback for each ESM and ISM
>> Output: `data/ensemble/{bmp}_{year0}_{year1}.nc`

#### Figures for draft
> Draftplot_ts_control: Plot basin temperatures for CTRL and CMIP6 including reanalysis range
>> Output: `draftfigs/ts_control.png`
> Draftplot_single_ensemble: Plot temp, ice mass loss and SLR for each basin for a single ensemble memeber
>> Output: `draftfigs/single_ensemble.png`
> Draftplot_masks: Plot calving and basal melt maps for old and new version
>> Output: `draftfigs/masks.png`
> Draftplot_ORF: Plot ocean response functions for 5 regional perturbations
>> Output: `draftfigs/all_orfs.png`
> Draftplot_linearitycheck: Plot basin temperature perturbations for comparible experiments as linearity check
>> Output: `draftfigs/linearitycheck.png`
> Draftplot_full_ensemble: Plot sea level projections for 3 SSPS and 2 basal melt params
>> Output: `draftfigs/full_ensemble_{year0}_{year1}.png`

#### Unnneccesary notebooks 
> Video_subsurfacetemp: Make a video of a temperature map at depth and time series of basin temperatures
> Plot_perturbations: Plot monthly time series of basin temp and temp anomalies for various runs
> Plot_basinmap_eveline: Make separate map of ocean basins
