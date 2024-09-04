### 此檔案繪製給定日期之天氣圖，與選定氣象變數之重要性詮釋

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
plt.rcParams['font.family'] = 'Noto Sans CJK TC'

### Read the data
lon = np.arange(110, 127.1, 0.25)
lat = np.arange(30, 9.9, -0.25)
lev = np.array([200, 500, 700, 850, 1000])

target_date = 20050502  #TODO
dates = np.loadtxt('./Data/FTdate_56_f1980.txt', dtype=int)
target_date_i = np.where(dates == target_date)[0]
X = np.load('./Data/vars_hr_f1980.npz')['var'][target_date_i]
X[:,0,:,:]  *= 1e5
X[:,-1,:,:] *= 1e2

shap_sample = np.loadtxt('./Data/shap_samples.txt', dtype=int)
target_date_i = np.where(dates == target_date)[0]
target_date_j = np.where(shap_sample == target_date_i)[0]
shap = np.load('./Data/shap_f1980.npz')['shap'][target_date_j].squeeze()


### 繪圖變數指定 & 標題文字處理
th = [40, 80, 145]
ticks = ['<%imm'%th[0], '%i-%imm'%(th[0], th[1]), '%i-%imm'%(th[1], th[2]), '>%imm'%th[2]]
vname = ['D', 'Z', 'T', 'U', 'V', 'W', 'PW']
vname_zh = ['輻散場 [$10^5$ $s^{-1}$]', '高度場 [gpm]', '溫度場 [K]', '緯向風速 [m/s]', '經向風速 [m/s]', '上升速度 [m/s]', '可降水量 [mm]']
title = []
title_full = []
for i in range(30):
    title.append(f'{vname[i//5]}{lev[i%5]}')
    title_full.append(f'{lev[i%5]}hPa {vname_zh[i//5]}')
title.append("PW")
title_full.append(vname_zh[-1])
var = lambda string: X[:, title.index(string), :, :].squeeze()


### Plotting
fig, ax = plt.subplots(1,2, figsize=(10,6), dpi=200)

### Coastline & grid
def create_basemap(ax, lon, lat):
    m = Basemap(projection='cyl',
                llcrnrlon=np.min(lon), llcrnrlat=np.min(lat),
                urcrnrlon=np.max(lon), urcrnrlat=np.max(lat), resolution='l', ax=ax)
    m.drawparallels(np.arange(10, 60, 5), labels=[1, 0, 0, 0], linewidth=0.3)
    m.drawmeridians(np.arange(90, 161, 5), labels=[0, 0, 0, 1], linewidth=0.3)
    m.drawcoastlines(linewidth=0.8)
    return m
m1 = create_basemap(ax[0], lon, lat)
m2 = create_basemap(ax[1], lon, lat)



### PW
CS = ax[0].contourf(lon, lat, var('PW'), levels=np.arange(36, 67, 2),
                  cmap="Blues", extend='both', alpha=0.9)
plt.colorbar(CS, orientation='vertical', fraction=0.033)


### Temperature
# CS = ax[0].contourf(lon, lat, var('T850'), 
#                     levels = np.arange(286, 296.1, 1),
#                   cmap='RdBu_r', extend='both', alpha=0.72)
# plt.colorbar(CS, orientation='vertical',
#                 fraction=0.033, extend='max')


### GP Height
CS = ax[0].contour(lon, lat, var('Z850')/9.8, levels=np.arange(100, 2000, 20),
                   colors='black', linewidths=1.5)
# CS = ax[0].contour(lon, lat, var('Z850'), levels=[1380,1500], colors='black',linewidths=1.8)
plt.clabel(CS, fontsize=10)


### Divergence
# CS = ax[0].contourf(lon, lat, var('W700'), levels=np.arange(-8, 8.1, 1),
#                   cmap='RdBu_r', extend='both', alpha=0.7)
# plt.colorbar(CS, orientation='vertical',
#               fraction=0.033, extend='max')


### Wind 
n = 4
if (lon.ndim == 1): lon, lat = np.meshgrid(lon, lat)
Q = ax[0].quiver(lon[::n,::n], lat[::n,::n], var('U1000')[::n,::n], var('V1000')[::n,::n], 
          color='#666', scale=160)
ax[0].quiverkey(Q, 0.88, 1.08, 7, label='7 m/s', 
              fontproperties={'weight':'bold'}, labelpos='E')

### SHAP
ch = title.index('U1000')
Class = 3 #TODO
CS = ax[1].pcolormesh(lon, lat, shap[Class, :, :, ch], cmap='coolwarm',
                      vmin=-0.001, vmax=0.001)
plt.colorbar(CS, orientation='vertical', fraction=0.033, extend='both')


### Title
title_date = f"{str(target_date)[:4]}.{str(target_date)[4:6]}.{str(target_date)[6:]}"
ax[1].set_title(f'{ticks[Class]} Feature Attribution Map [{title_date}]\n'
                f'{title_full[ch]}', loc='left', fontsize=11)

ax[0].set_title(f'ERA5 Reanalysis [{title_date}]\n'
                '850 hPa Wind,  Z(contour) [gpm],  PW(shading) [mm]',
                loc='left', fontsize=10)
plt.tight_layout()

#%%

### 繪製選定日期之氣象參數貢獻度長條圖
n = 8
attri_ch = np.sum(shap[Class], axis=(0,1))
top_n_indices = np.argsort(abs(attri_ch))[-n:] 
top_n_values = attri_ch[top_n_indices] 
top_n_titles = [title[i] for i in top_n_indices]


plt.figure(dpi=200)
plt.bar(top_n_titles, top_n_values, width=0.5)
plt.xlabel('Variables')
plt.ylabel('Values')
plt.title(f'{title_date}\n'
          f'{ticks[Class]} Feature Attribution', loc='left', fontsize=11)
plt.axhline(y=0, color='k', linestyle='dotted', alpha=0.4)
plt.grid(axis='x', alpha=0.5)
plt.tight_layout()
