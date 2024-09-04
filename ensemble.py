import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

### Read the data
lon = np.arange(110, 127.1, 0.25)
lat = np.arange(30, 9.9, -0.25)
lev = np.array([200, 500, 700, 850, 1000])

yt, yp = 3, 3  # TODO
true, pred = np.loadtxt('./Data/Y_true_pred_f1980.txt', skiprows=1, unpack=True)
mask = (true==yt) & (pred==yp)
X = np.load('./Data/vars_hr_f1980.npz')['var'][mask]
dates = np.loadtxt('./Data/FTdate_56_f1980.txt', dtype=str)[mask]

X = np.mean(X, 0)
X[0,:,:]  *= 1e5
X[-1,:,:] *= 1e2


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
var = lambda string: X[title.index(string), :, :].squeeze()



plt.figure(figsize=(9,7), dpi=200)

### Coastline & grid
m = Basemap(projection='cyl',
            llcrnrlon=np.min(lon), llcrnrlat=np.min(lat),
            urcrnrlon=np.max(lon), urcrnrlat=np.max(lat),resolution='l')
m.drawparallels(np.arange(10,60,5),labels=[1,0,0,0], linewidth=0.3)
m.drawmeridians(np.arange(90,161,5),labels=[0,0,0,1], linewidth=0.3)
m.drawcoastlines(linewidth=0.8)


### PW
CS = plt.contourf(lon, lat, var('PW'), levels=np.arange(36, 67, 2),
                  cmap="Blues", extend='both', alpha=1)
plt.axis('scaled')
plt.colorbar(CS,orientation='vertical', fraction=0.024)


###  Temperature
# CS = plt.contourf(lon, lat, var('T850'), levels = np.arange(286, 296.1, 1),
#                   cmap='RdBu_r', extend='both', alpha=0.72)
# plt.colorbar(CS,orientation='vertical',
#                 fraction=0.024, extend='max')


### GP Height
CS = plt.contour(lon, lat, var('Z850')/9.8, levels=np.arange(100, 1500, 10),
                  colors='black',linewidths=1.5)
# CS = plt.contour(lon,lat,z850, levels=[1380,1500], colors='black',linewidths=1.8)
plt.clabel(CS, fontsize=10)


### Divergence
# CS = plt.contourf(lon, lat, var('D200'), levels=np.arange(-8, 9, 1)/4,
#                   cmap='RdBu_r', extend='both', alpha=0.7)
# plt.colorbar(CS,orientation='vertical', fraction=0.024)


### Wind 
n = 4
if (lon.ndim == 1): lon, lat = np.meshgrid(lon, lat)
Q = plt.quiver(lon[::n,::n], lat[::n,::n], var('U850')[::n,::n], var('V850')[::n,::n], 
          color='#BBB', scale=160)
plt.quiverkey(Q, 1.06, 0.91, 7, label='7 m/s', 
              fontproperties={'weight':'bold'}, labelpos='E')



plt.title(f'[{len(dates)}#]\n預測 {ticks[yp]}｜實際 {ticks[yt]}',
          loc='left', fontsize=11)

plt.title('850hPa  ERA5 Reanalysis\n'
          'Wind[m/s], Z[gpm], PW[mm]', loc='right', fontsize=10)
