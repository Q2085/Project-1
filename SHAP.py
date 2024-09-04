import netCDF4 as nc
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
from mpl_toolkits.basemap import Basemap
plt.rcParams['font.family'] = 'Noto Sans CJK TC'


### Read the SHAP values
shap = np.load('shap_f1980.npz')['shap']
shap_sample = np.loadtxt('shap_samples.txt', dtype=int)
true, pred = np.loadtxt('Y_true_pred_f1980.txt', skiprows=1, unpack=True)[:, shap_sample]

lon = np.arange(110, 127.1, 0.25)
lat = np.arange(30, 9.9, -0.25)
lev = np.array([200, 500, 700, 850, 1000])


### Plot the attribution bar charts
yt, yp = 3, 3 # TODO
mask = (true==yt) & (pred==yp)
dates = np.loadtxt('FTdate_56_f1980.txt', dtype=str)
dates = dates[shap_sample][mask]

Class = yt
attri_ch = np.sum((shap[mask][:, Class, :, :, :]), axis=(0,1,2))

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

n = 8
top_n_indices = np.argsort(abs(attri_ch))[-n:] 
top_n_values = attri_ch[top_n_indices] 
top_n_titles = [title[i] for i in top_n_indices]


plt.figure(dpi=200)
plt.bar(top_n_titles, top_n_values, width=0.5)
plt.xlabel('Variables')
plt.ylabel('Values')
plt.title(f'預測 {ticks[yp]}｜實際 {ticks[yt]}｜[{len(dates)}#]\n'
          f'{ticks[Class]} Feature Attributions', loc='left', fontsize=11)
plt.axhline(y=0, color='k', linestyle='dotted', alpha=0.4)
plt.grid(axis='x', alpha=0.5)
plt.tight_layout()


### Coastline & grid
fig, ax = plt.subplots(1,2, figsize=(11,6),dpi=150)
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


### Plotting SHAP & specified varable
ch = title.index('Z1000')  # TODO
X = np.load('vars_hr_f1980.npz')['var'][shap_sample]
X[:,0,:,:]  *= 1e5
X[:,-1,:,:] *= 1e2
X[:,5:10,:,:] /= 9.8
CS = ax[0].contourf(lon, lat, np.mean(X[mask], 0)[ch,:,:], cmap='RdBu_r',
                    levels=np.arange(40, 121, 5), extend='both',
                    # levels=np.arange(-12, 13, 2), extend='both',
                    # levels=np.arange(0.3, 0.7, 0.02), extend='both'
                    # levels=np.arange(-3.0, 3.01, 0.5), extend='both'
                    # levels=np.arange(36, 67, 2), extend='both'
                    # levels = np.arange(286, 296.1, 1), extend='both'
                    )
plt.colorbar(CS, shrink=0.5)
ax[0].set_title(f'預測 {ticks[yp]}｜實際 {ticks[yt]}｜[{len(dates)}#]\n'
                f'{title_full[ch]}', loc='left')

x = np.sum(shap[mask][:, Class, :, :, :], axis=0)
vmax = np.quantile(abs(x), 0.992)
CS = ax[1].pcolormesh(lon, lat, x[:,:,ch], cmap='coolwarm',
                 vmax=vmax, vmin=-vmax)

ax[1].set_title(f'{ticks[Class]} Feature Attributions\n'
                f'{title_full[ch]}', loc='left')
plt.colorbar(CS, shrink=0.5)
plt.tight_layout()

#%%

### Plot TCCIP Rainfall
t0 = (datetime(1980, 1, 1) - datetime(1960, 1, 1)).days
dates = np.loadtxt('FTdate_56_f1980.txt', dtype=str)
dates = np.array([datetime.strptime(date_str, "%Y%m%d") # + timedelta(days=1) #TODO
                  for date_str in dates]) 

days = (dates - datetime(1980, 1, 1)).astype('timedelta64[D]').astype(int)
dates = dates[shap_sample][mask]
prec = nc.Dataset("TCCIP_GriddedRain_TWN.nc").variables['PREC_5km'][t0:][days][shap_sample][mask]


fig = plt.figure(figsize=(5,8), dpi=200)
color = ['#e6e6e6', '#a0fffa','#00cdff','#0096ff','#0069ff',
         '#339911','#69e933','#ffee44','#ffc800',
         '#ff9600','#f44400','#c80000','#9d0000',
         '#aa229b','#c800d2','#fa11f9','#ff64ff', '#ffc8ff']
cmap = mcolor.ListedColormap(color)
cmap.set_over(color[-1])
bounds = np.array([0,1,2,4,6,10,15,20,30,40,50,70,90,110,130,150,200])
norm = mcolor.BoundaryNorm(bounds, ncolors=len(color)-1)

m = Basemap(projection='cyl',
            llcrnrlon=119.9, llcrnrlat=21.8,
            urcrnrlon=122.1, urcrnrlat=25.4, resolution='l')
m.drawparallels(np.arange(22, 25.31, 0.5), labels=[1,0,0,0], linewidth=0.2)
m.drawmeridians(np.arange(119,122.51,0.5), labels=[0,0,0,1], linewidth=0.2)
m.drawcoastlines(linewidth=0.8)

topo= nc.Dataset("TOPO.nc").variables['TOPO'][:]*100
lon_, lat_ = np.meshgrid(np.linspace(118.6387,123.3957,1024), np.linspace(21.21948,25.97643,1024))
plt.contourf(lon_, lat_, topo, cmap='Greys')


sample = 4  # TODO
# sample = np.where(dates == datetime.strptime('19830602', '%Y%m%d'))[0][0]
sample_date = datetime.strftime(dates[sample], "%Y.%m.%d")

lon_, lat_ = np.meshgrid(np.linspace(120,122,41), np.linspace(21.9,25.3,69))

CS = plt.pcolormesh(lon_, lat_, prec[sample],
                    cmap=cmap, norm=norm, alpha=0.6)
plt.colorbar(CS, orientation='vertical', ticks=bounds, shrink=0.6, extend='max')
plt.title(f'{sample_date}  日累積雨量 [mm]', loc='left', fontweight='400', fontsize=11.5)
plt.tight_layout()
