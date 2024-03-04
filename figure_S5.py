#!/usr/bin/env python3.8

import numpy as np
from netCDF4 import Dataset

from scipy import stats
import matplotlib.pyplot as plt
import mpl_toolkits.basemap as bm
from matplotlib.patches import Rectangle

dir1 = "Data path"
dir2 = "Data path"
dir3 = "Data path"

f = Dataset(dir1+'filename')
g = Dataset(dir2+'filename')

h = Dataset(dir3+'filename')



# t2m
timevar = f.variables['time']
lonvar = f.variables['lon'][:]
latvar = f.variables['lat'][:]
tas = f.variables['t2m'][:,:,:]
tas=tas-273.15
mtas=np.mean(tas,axis=0)

anom=[]
anom=tas-mtas
mj1 = anom
# sst
timevar1 = g.variables['time']
lonvar1 = g.variables['lon'][:]
latvar = g.variables['lat'][:]
sst = g.variables['sst'][:,:,:]

msst=np.mean(sst,axis=0)

anom1=[]
anom1=sst-msst
mj2 = anom1

# hgt
timevar2 = h.variables['time']
lonvar2 = h.variables['lon'][:]
latvar2 = h.variables['lat'][:]
levvar2 = h.variables['level'][:]
hgt = h.variables['z'][:,4,:,:]
hgt=hgt/10
mhgt=np.mean(hgt,axis=0)
anom2=[]
anom2=hgt-mhgt
mj3 = anom2


nt, nlat, nlon = mj1.shape
ngrd = nlon*nlat
#here we are defining the index
lonE=65
lonW=80
latS=31
latN=39

lon1=np.where(lonvar==lonE)
lon2=np.where(lonvar==lonW)
lat1=np.where(latvar==latS)
lat2=np.where(latvar==latN)
print(lon1,lon2,lat1,lat2)

anom=mj1
#anom=djfsst-msst
#print(anom)
#calculate the anomaly for the t2m dataset
panom=[]
panom=mj1
#sst

panom_sst=[]
panom_sst=mj2
#calculate the anomaly for the hgt dataset
panom_hgt=[]
panom_hgt=mj3

lonx, latx = np.meshgrid(lonvar, latvar)
weights = np.cos(latx * np.pi / 180.)

ts_avg = np.zeros(nt)

for it in np.arange(nt):
    ts_avg[it] = np.ma.average(anom[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])

#here you can define your standard deviation
std_ts_avg=round(np.std(ts_avg),2)
print(std_ts_avg)
#here you check your astiN/satiP years values 
satiN_val= ts_avg[ts_avg<=-1*(std_ts_avg)]
satiP_val= ts_avg[ts_avg>=(std_ts_avg)]
# print(satiP_val['time'])
# print('..........')
#print(satiN_val.shape,satiP_val.shape)

lan_amp=np.mean(np.sqrt(satiN_val*satiN_val))
meaneln=np.mean(satiP_val)
t2m_satiP = panom[ts_avg>=round(std_ts_avg,2), :,:] 
t2m_satiN = panom[ts_avg<=round(std_ts_avg,2)*-1., :,:]
#####Next the SST composite is defined###
satiP_mean= np.mean(t2m_satiP,axis=0)
satiN_mean= np.mean(t2m_satiN,axis=0)

#-----sst
# #-----for hgt
sst_satiP = panom_sst[ts_avg>=round(std_ts_avg,2), :,:] 
sst_satiN = panom_sst[ts_avg<=round(std_ts_avg,2)*-1., :,:]
satiP_sst_mean= np.mean(sst_satiP,axis=0)
satiN_sst_mean= np.mean(sst_satiN,axis=0)

# #-----for hgt
hgt_satiP = panom_hgt[ts_avg>=round(std_ts_avg,2), :,:] 
hgt_satiN = panom_hgt[ts_avg<=round(std_ts_avg,2)*-1., :,:]
satiP_hgt_mean= np.mean(hgt_satiP,axis=0)
satiN_hgt_mean= np.mean(hgt_satiN,axis=0)

#---------------------SATIp----------------------------------------------------
df=len(satiP_val)-2
#print(df)
numt=[]
numt=satiP_mean
#print(numt.shape)
denmt=[]
denmt=np.sqrt(np.mean(pow(t2m_satiP,2)))/np.sqrt(df)
#print(denmt.shape)
tscore = []
tscore = abs(numt/denmt)
#print(tscore)
#tscore1 = tscore

t80 = stats.t.ppf(1-0.10, df)
t90 = stats.t.ppf(1-0.05, df)
print(t80)
print(t90)

fig = plt.figure(figsize=(12, 13), constrained_layout=True)
ax1 = fig.add_subplot(3, 2, 1)
lons, lats = np.meshgrid(lonvar, latvar)
m = bm.Basemap(projection='cyl',llcrnrlat=20, llcrnrlon=55,urcrnrlat=41, urcrnrlon=91, lon_0=-80, lat_0=0, resolution='l')
x, y = m(lons, lats)
m.drawcoastlines(linewidth=0.25)
m.drawmapboundary(fill_color='white')

m.drawcountries(linewidth=0.25)

parallels = np.arange(20,41,5.) # make latitude lines
meridians = np.arange(55,91,5.) # make longitude lines
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.01)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,linewidth=0.01)
map2=m.contourf(x,y,satiP_mean,np.arange(-1.6,1.8,0.2), cmap=plt.cm.RdBu_r, extend='both')
plt.gca().add_patch(Rectangle((65, 31),
                        15, 8,
                        fc ='none', 
                        ec ='black',
                        lw = 2))
plt.contourf(x,y,tscore[:,:], levels=[-1*t90, -1*t80, t80, t90],extend='both',
              colors = 'none', hatches=['..',None,None, None, '..'],alpha=0)

plt.title("a) TMP(SATIp)", loc='left', fontsize=14 )
plt.title("ERA5", loc='right', fontsize=14 )
#-----------------------SATIn--------------------------------------------------
df=len(satiN_val)-2
#print(df)
numt=[]
numt=satiN_mean
#print(numt.shape)
denmt=[]
denmt=np.sqrt(np.mean(pow(t2m_satiN,2)))/np.sqrt(df)
#print(denmt.shape)
tscore = []
tscore = abs(numt/denmt)

t80 = stats.t.ppf(1-0.10, df)
t90 = stats.t.ppf(1-0.05, df)
print(t80)
print(t90)
#--------------------------------------------------
ax1 = fig.add_subplot(3, 2, 2)
lons, lats = np.meshgrid(lonvar, latvar)
m = bm.Basemap(projection='cyl',llcrnrlat=20, llcrnrlon=55,urcrnrlat=41, urcrnrlon=91, lon_0=-80, lat_0=0, resolution='l')
x, y = m(lons, lats)
m.drawcoastlines(linewidth=0.25)
m.drawmapboundary(fill_color='white')
m.drawcountries(linewidth=0.25)
parallels = np.arange(20,41,5.) # make latitude lines
meridians = np.arange(55,91,5.) # make longitude lines
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.01)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,linewidth=0.01)
map3 = m.contourf(x,y,satiN_mean,np.arange(-1.6,1.8,0.2),cmap=plt.cm.RdBu_r,extend='both')
plt.gca().add_patch(Rectangle((65, 31),
                        15, 8,
                        fc ='none', 
                        ec ='black',
                        lw = 2))
plt.contourf(x,y,tscore[:,:], levels=[-1*t90, -1*t80, t80, t90],extend='both',
              colors = 'none', hatches=['..',None,None, None, '..'],alpha=0)
plt.title("d) TMP(SATIn)", loc='left', fontsize=14 )
plt.title("ERA5", loc='right', fontsize=14 )
cbar_ax2 = fig.add_axes([0.2, 0.64, 0.6, 0.03])  # [left, bottom, width, height]
cbar2 = fig.colorbar(map3, cax=cbar_ax2, orientation='horizontal', shrink=0.8, pad=0.8)
cbar2.ax.tick_params(labelsize=14) 
#------------------------ Z200 Composites SATp
ax1 = fig.add_subplot(3, 2, 3)
df =len(satiP_val)-2
#print(df)
numt=[]
numt=satiP_hgt_mean
#print(numt.shape)
denmt=[]
denmt=np.sqrt(np.mean(pow(hgt_satiP,2)))/np.sqrt(df)
#print(denmt.shape)
tscore = []
tscore = abs(numt/denmt)
#print(tscore)
t80 = stats.t.ppf(1-0.10, df)
t90 = stats.t.ppf(1-0.05, df)


lons, lats = np.meshgrid(lonvar, latvar)
m = bm.Basemap(projection='cyl',llcrnrlat=10, llcrnrlon=10,urcrnrlat=61, urcrnrlon=121, lon_0=-80, lat_0=0, resolution='l')
x, y = m(lons, lats)
m.drawcoastlines(linewidth=0.25)		
m.drawcountries(linewidth=0.25)	

parallels = np.arange(10,61,10.) # make latitude lines
meridians = np.arange(10,121,20.) # make longitude lines
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.01)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,linewidth=0.01)
map3=m.contourf(x,y,satiP_hgt_mean,np.arange(-35,36,5),cmap=plt.cm.RdBu_r,extend='both')

plt.contourf(x,y,tscore[:,:], levels=[-1*t90, -1*t80, t80, t90],extend='both',
              colors = 'none', hatches=['..',None,None, None, '..'],alpha=0)
plt.title("b) Z200(SATIp)", loc='left', fontsize=14 )
plt.title("ERA5", loc='right', fontsize=14 )


#------------------------ Z200 Composites SATn---------------
ax3 = fig.add_subplot(3, 2, 4)
df =len(satiN_val)-2
#print(df)
numt=[]
numt=satiN_hgt_mean
#print(numt.shape)
denmt=[]
denmt=np.sqrt(np.mean(pow(hgt_satiN,2)))/np.sqrt(df)
tscore = []
tscore = abs(numt/denmt)
#print(tscore)
t80 = stats.t.ppf(1-0.10, df)
t90 = stats.t.ppf(1-0.05, df)


lons, lats = np.meshgrid(lonvar, latvar)
m = bm.Basemap(projection='cyl',llcrnrlat=10, llcrnrlon=10,urcrnrlat=61, urcrnrlon=121, lon_0=-80, lat_0=0, resolution='l')
x, y = m(lons, lats)
m.drawcoastlines(linewidth=0.25)		# Draw coastal boundaries
m.drawcountries(linewidth=0.25)	
parallels = np.arange(10,61,10.) # make latitude lines
meridians = np.arange(10,121,20.) # make longitude lines
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.01)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,linewidth=0.01)
map4=m.contourf(x,y,satiN_hgt_mean,np.arange(-35,36,5),cmap=plt.cm.RdBu_r,extend='both')
plt.contourf(x,y,tscore[:,:], levels=[-1*t90, -1*t80, t80, t90],extend='both',
              colors = 'none', hatches=['..',None,None, None, '..'],alpha=0)
plt.title("e) Z200(SATIn)", loc='left', fontsize=12 )
plt.title("ERA5", loc='right', fontsize=12 )

# plt.subplots_adjust(left=0.1,
#                     bottom=0.1, 
#                     right=0.95, 
#                     top=0.9, 
#                     wspace=0.1, 
#                     hspace=0.3)
cbar_ax4 = fig.add_axes([0.2, 0.33, 0.6, 0.03])  # [left, bottom, width, height]
cbar4 = fig.colorbar(map4, cax=cbar_ax4, orientation='horizontal', shrink=0.8, pad=0.8)
cbar4.ax.tick_params(labelsize=14) 

#-----------sst
#---------------------SATIp----------------------------------------------------
df=len(satiP_val)-2
#print(df)
numt=[]
numt=satiP_sst_mean
#print(numt.shape)
denmt=[]
denmt=np.sqrt(np.mean(pow(sst_satiP,2)))/np.sqrt(df)
#print(denmt.shape)
tscore = []
tscore = abs(numt/denmt)
#print(tscore)
#tscore1 = tscore

t80 = stats.t.ppf(1-0.10, df)
t90 = stats.t.ppf(1-0.05, df)
print(t80)
print(t90)

ax4 = fig.add_subplot(3, 2, 5)
#def composites(sample):
lons, lats = np.meshgrid(lonvar, latvar)
m = bm.Basemap(projection='cyl',llcrnrlat=-40, llcrnrlon=0,urcrnrlat=41, urcrnrlon=361, lon_0=-80, lat_0=0, resolution='l')
x, y = m(lons, lats)
m.drawcoastlines(linewidth=0.25)
m.drawmapboundary(fill_color='white')
m.fillcontinents(color='gray',lake_color='gray',zorder=2)
parallels = np.arange(-40,41,20.) # make latitude lines
meridians = np.arange(-180,181,60.) # make longitude lines
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.01)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,linewidth=0.01)
map2=m.contourf(x,y,satiP_sst_mean,np.arange(-0.7,0.8,0.1),cmap=plt.cm.RdBu_r,extend='both')

plt.contourf(x,y,tscore[:,:], levels=[-1*t90, -1*t80, t80, t90],extend='both',
              colors = 'none', hatches=['...',None,None, None, '...'],alpha=0)
plt.title("c) SST(SATIp)", loc='left', fontsize=14 )
plt.title("ERA5", loc='right', fontsize=14 )
#-----------------------SATIn--------------------------------------------------

df=len(satiN_val)-2
#print(df)
numt=[]
numt=satiN_sst_mean
#print(numt.shape)
denmt=[]
denmt=np.sqrt(np.mean(pow(sst_satiN,2)))/np.sqrt(df)
#print(denmt.shape)
tscore = []
tscore = abs(numt/denmt)
#print(tscore)
#tscore1 = tscore

t80 = stats.t.ppf(1-0.10, df)
t90 = stats.t.ppf(1-0.05, df)


ax5 = fig.add_subplot(3, 2, 6)


lons, lats = np.meshgrid(lonvar, latvar)
m = bm.Basemap(projection='cyl',llcrnrlat=-40, llcrnrlon=0,urcrnrlat=41, urcrnrlon=361, lon_0=-80, lat_0=0, resolution='l')
x, y = m(lons, lats)
m.drawcoastlines(linewidth=0.25)
m.drawmapboundary(fill_color='white')
m.fillcontinents(color='gray',lake_color='gray',zorder=2)
parallels = np.arange(-40,41,20.) # make latitude lines
meridians = np.arange(-180,181,60.) # make longitude lines
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.01)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,linewidth=0.01)
map3=m.contourf(x,y,satiN_sst_mean,np.arange(-0.7,0.8,0.1),cmap=plt.cm.RdBu_r,extend='both')

plt.contourf(x,y,tscore[:,:], levels=[-1*t90, -1*t80, t80, t90],extend='both',
              colors = 'none', hatches=['...',None,None, None, '...'],alpha=0)

plt.title("f) SST(SATIn)", loc='left', fontsize=14 )
plt.title("ERA5", loc='right', fontsize=14 )
# Adjust layout to make space for colorbars
# plt.subplots_adjust(left=0.1,
#                     bottom=0.1, 
#                     right=0.95, 
#                     top=0.9, 
#                     wspace=0.1, 
#                     hspace=0.3)
fig.tight_layout()

cbar_ax5 = fig.add_axes([0.2, 0.08, 0.6, 0.03])  # [left, bottom, width, height]
cbar5 = fig.colorbar(map3, cax=cbar_ax5, orientation='horizontal', shrink=0.8, pad=0.8)
cbar5.ax.tick_params(labelsize=14) 
# cbar_ax2 = fig.add_axes([0.2, 0.23, 0.6, 0.03])  # [left, bottom, width, height]
# cbar2 = fig.colorbar(map2, cax=cbar_ax2, orientation='horizontal', shrink=0.8, pad=0.8)
# cbar2.ax.tick_params(labelsize=14) 
plt.savefig('figure_S5_adbecf.png', dpi = 300)
plt.show()
