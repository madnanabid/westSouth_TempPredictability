import numpy as np
from netCDF4 import Dataset
from scipy import stats


import matplotlib.pyplot as plt
import mpl_toolkits.basemap as bm
from matplotlib.patches import Rectangle

f = Dataset('Data path + file')
g = Dataset('Data path + file')

lonE=190
lonW=240
latS=-5
latN=5

timevar = f.variables['time']
lonvar = f.variables['lon'][:]
latvar = f.variables['lat'][:]
tas = f.variables['sst'][:,:,:]

nt, nlat, nlon = tas.shape
ngrd = nlon*nlat
print(nt)
print(ngrd)

#print(tas)
#this averages over time
msst=np.mean(tas,axis=0)
#print(msst)
#print(msst.shape)
anom=[]
anom=tas-msst
#print(anom.shape)

#here we are defining the index

lon1=np.where(lonvar==lonE)
lon2=np.where(lonvar==lonW)
lat1=np.where(latvar==latS)
lat2=np.where(latvar==latN)
print(lon1,lon2,lat1,lat2)

lonx, latx = np.meshgrid(lonvar, latvar)
weights = np.cos(latx * np.pi / 180.)

tas_avg = np.zeros(nt)

for it in np.arange(nt):
    tas_avg[it] = np.ma.average(anom[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])

tas_avg = tas_avg /np.std(tas_avg)

#--------------------------------------------------------
timevar2 = g.variables['time']
lonvar2 = g.variables['lon'][:]
latvar2 = g.variables['lat'][:]
tas2 = g.variables['t2m'][:,:,:]
#print(tas)
#this averages over time
msst2=np.mean(tas2,axis=0)
#print(msst)
#print(msst.shape)
#anom=[]
anom2=tas2-msst2

slopes = np.zeros((anom.shape[1],anom.shape[2]))
p_values = np.zeros((anom.shape[1],anom.shape[2]))
corrs = np.zeros((anom.shape[1],anom.shape[2]))



for x in range(0, anom2.shape[1]):
  for y in range(0, anom2.shape[2]):
    punto = anom2[:,x,y]
#    slope, intercept, r_value, p_value, std_err = stats.linregress(ts_nino34,punto)
    slope, intercept, r_value, p_value, std_err = stats.linregress(tas_avg,punto)
    slopes[x,y] = slope
    p_values[x,y] = p_value
    corrs[x,y] = r_value

lonEW=65 ; lonWW=80; latSW=31; latNW=39
lon3=np.where(lonvar==lonEW)
lon4=np.where(lonvar==lonWW)
lat3=np.where(latvar==latSW)
lat4=np.where(latvar==latNW)

print(lon3,lon4,lat3,lat4)
val = []
val = np.ma.average(slopes[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
print(np.round(val,2))

#print(slope)
#print(p_value)

#--------------------------------
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(2, 2, 1)
#domain definition plotting
lat1=20
lat2=41
lon1=55
lon2=91
[lonall, latall] = np.meshgrid(lonvar2[:], latvar2[:])
mapproj = bm.Basemap(projection='cyl',llcrnrlat=lat1, llcrnrlon=lon1,urcrnrlat=lat2, urcrnrlon=lon2, lon_0=-180, lat_0=0, resolution='l')
mapproj.drawcoastlines(linewidth=0.25)
mapproj.drawcountries(linewidth=0.25)
x, y = mapproj(lonall, latall)
levels =[-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.8,1.2]
ax1=plt.contourf(x,y,slopes[:,:],np.arange(-1.2,1.3,0.1), cmap=plt.cm.RdBu_r, extend = 'both')
# Add the rectangular box to the plot
plt.gca().add_patch(Rectangle((65, 31),
                        15, 8,
                        fc ='none', 
                        ec ='black',
                        lw = 2))
####  T test for statistical signifcance test ######
df =len(tas_avg)-2
#print(df)
numt=[]
numt=corrs[:,:]*np.sqrt(df)
denmt=[]
denmt=np.sqrt(1-pow(corrs,2))
tscore = []

tscore = (numt/denmt)

t90 = stats.t.ppf(1-0.10, df)
t95 = stats.t.ppf(1-0.05,df)
print(t90)
print(t95)

plt.contourf(x,y,tscore[:,:], levels=[-1*t95, -1*t90, t90, t95],extend='both',
        colors = 'none', hatches=['..',None,None, None, '..'],alpha=0)

parallels = np.arange(20,41,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(55,91,5.) # make longitude lines every 5 degrees from 95W to 70W
mapproj.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.1)
mapproj.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,linewidth=0.1)


plt.title("c) Reg.(NINO34,SAT)", loc='left', fontsize=14)
plt.title("ERA5", loc='right', fontsize=14)

#------------------------NINO34 vs SAT SEAS5

dir1= 'Data Path'
dir2= 'Data Path'

bslope = np.zeros((25,181,360))
bcorrs = np.zeros((25,181,360))


count=0
for i in range(25):
#    print(i)
    i+=1
    print(i)
    m2 = Dataset(dir1+'sys5_e'+str(i)+'.nc')
    d1 = Dataset(dir2+'sys5_e'+str(i)+'.nc')
    time = m2.variables['time']
    lon = m2.variables['lon'][:]
    lat = m2.variables['lat'][:]
    may = m2.variables['m1'][:,:,:]
    jun = m2.variables['m2'][:,:,:]
    mj= (may+jun)/2
    nt, nlat, nlon = mj.shape
    ngrd = nlon*nlat

    lonT = d1.variables['lon'][:]
    latT = d1.variables['lat'][:]
    may1 = d1.variables['m1'][:,:,:]
    jun1 = d1.variables['m2'][:,:,:]
    mj1= (may1+jun1)/2

    print(mj1.shape)


    lonE=190 ; lonW=240; latS=-5; latN=5
    lon1=np.where(lonT==lonE)
    lon2=np.where(lonT==lonW)
    lat1=np.where(latT==latS)
    lat2=np.where(latT==latN)
    msst=np.mean(mj1,axis=0)
    print(lon1,lon2,lat1,lat2)
#    print()
#    print(msst.shape)

    anom=[]
    anom = mj1-msst

    lonx, latx = np.meshgrid(lon, lat)
    weights = np.cos(latx * np.pi / 180.)
    #print(weights.shape)
    tas_avg = np.zeros(nt)

    for it in np.arange(nt):
        tas_avg[it] = np.ma.average(anom[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    
    tas_avg = tas_avg /np.std(tas_avg)
    
    panom=[]
    panom =(may+jun)/2-np.mean((may+jun)/2,axis=0)
    
    slopes = np.zeros((anom.shape[1],anom.shape[2]))
    #print(slopes.shape)


    p_values = np.zeros((anom.shape[1],anom.shape[2]))
    corrs = np.zeros((anom.shape[1],anom.shape[2]))
    for x in range(0, panom.shape[1]):
        for y in range(0, panom.shape[2]):
            punto = panom[:,x,y]
            slope, intercept, r_value, p_value, std_err = stats.linregress(tas_avg,punto)
            slopes[x,y] = slope
            bslope[count,x,y] = slope
            bcorrs[count,x,y] = (0.5*(np.log(1.+r_value)-np.log(1.-r_value)))
            
    #print(slope)             
                    
    count+=1
    
    lonEW=65 ; lonWW=80; latSW=31; latNW=39
    lon3=np.where(lon==lonEW)
    lon4=np.where(lon==lonWW)
    lat3=np.where(lat==latSW)
    lat4=np.where(lat==latNW)
    #msst=np.mean(nd1,axis=0)
    print(lon3,lon4,lat3,lat4)
    val = []
    val = np.ma.average(slopes[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])], weights=weights[int(lat3[0]):int(lat4[0]),int(lon3[0]):int(lon4[0])])
    print(np.round(val,2))    
#print(bslope.shape)
bsst=np.mean(bslope,axis=0)
acorr=np.mean(bcorrs,axis=0)
#--------------------------------
ax2 = fig.add_subplot(2, 2, 2)
#domain for plotting
lat1=20
lat2=41
lon1=55
lon2=91
[lonall, latall] = np.meshgrid(lon[:], lat[:])
mapproj = bm.Basemap(projection='cyl',llcrnrlat=lat1, llcrnrlon=lon1,urcrnrlat=lat2, urcrnrlon=lon2, lon_0=-180, lat_0=0, resolution='l')
mapproj.drawcoastlines(linewidth=0.25)
mapproj.drawcountries(linewidth=0.25)
x, y = mapproj(lonall, latall)
levels =[-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.8,1.2]
ax2=plt.contourf(x,y,bsst[:,:],np.arange(-1.2,1.3,0.1), cmap=plt.cm.RdBu_r, extend = 'both')
# Add the rectangular box to the plot
plt.gca().add_patch(Rectangle((65, 31),
                        15, 8,
                        fc ='none', 
                        ec ='black',
                        lw = 2))
####  T test for statistical signifcance test ######
df =25-2
#print(df)
numt=[]
numt=acorr[:,:]*np.sqrt(df)
denmt=[]
denmt=np.sqrt(1-pow(acorr,2))
tscore = []

tscore = (numt/denmt)

t90 = stats.t.ppf(1-0.10, df)
t95 = stats.t.ppf(1-0.05,df)
print(t90)
print(t95)

plt.contourf(x,y,tscore[:,:], levels=[-1*t95, -1*t90, t90, t95],extend='both',
        colors = 'none', hatches=['..',None,None, None, '..'],alpha=0)

parallels = np.arange(20,41,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(55,91,5.) # make longitude lines every 5 degrees from 95W to 70W
mapproj.drawparallels(parallels,labels=[1,0,0,0],fontsize=10,linewidth=0.01)
mapproj.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10,linewidth=0.01)


plt.title("f) Reg.(NINO34,SAT)", loc='left', fontsize=14)
plt.title("SEAS5", loc='right', fontsize=14)
plt.tight_layout(pad=1.0,w_pad=2)
cbar_ax2 = fig.add_axes([0.2, 0.46, 0.6, 0.05])  # [left, bottom, width, height]
cbar2 = fig.colorbar(ax2, cax=cbar_ax2, orientation='horizontal', shrink=0.8, pad=0.8)
cbar2.ax.tick_params(labelsize=14) 

plt.savefig('figure5cf.png', dpi = 300)
