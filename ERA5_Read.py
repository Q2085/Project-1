### 此檔案將每日ERA5的資料合併為單一.npz檔案，以便後續讀取

import netCDF4 as nc
import numpy as np

rootgrp = nc.Dataset('./ERA5/2001050100.nc')

### 可以快速檢視變數名稱
# import re
# var = str(rootgrp.variables)
# str_var_name = re.findall(r'\blong_name.+', str(rootgrp.variables))

lon = rootgrp.variables['longitude'][:]
lat = rootgrp.variables['latitude'][:]
lev = rootgrp.variables['level'][:]

dates = np.loadtxt('FTdate_56_f1980.txt', dtype=str)
var = np.zeros((len(dates), 31, len(lat), len(lon)))

for i, date in enumerate(dates):
    
    rootgrp = nc.Dataset(f'./ERA5/{date}00.nc')
    
    # vor = rootgrp.variables['vo'][0, 1] 
    d = rootgrp.variables['d'][0] 
    z = rootgrp.variables['z'][0]
    q = rootgrp.variables['q'][0]
    IWV = np.trapz(rootgrp.variables['q'][0], [200,  500,  700,  850, 1000], axis=0) / 9.8
    T = rootgrp.variables['t'][0]
    u = rootgrp.variables['u'][0]
    v = rootgrp.variables['v'][0]    
    vo= rootgrp.variables['vo'][0]  
    w = rootgrp.variables['w'][0]

    var[i] = np.vstack([d, z, T, u, v, w, IWV[np.newaxis]])

np.savez('vars_hr_f1980.npz', var = var)
