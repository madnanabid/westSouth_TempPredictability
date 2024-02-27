# -*- coding: utf-8 -*-
"""
Created on Fri jun 23 09:11:04 2023

@author: Dr Irfan Rasheed
"""

import itertools

#
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pylab import *
import netCDF4 as nc
from scipy.stats import pearsonr
from scipy import stats
from scipy.stats import linregress
#from xclim import ensembles
#from netCDF4 import Datase
fig = plt.figure(figsize=(6,5))
#diri_output="./"
diri="data path"
#diri3="E:/DATA/ERA-DATA/ts/MJ/"
diri2="data path"
diri3="data path"
diri4="data path"
diri5="data path"

diri6="data path"
diri7="data path"

# diri6 = "E:/DATA/MERRA2/MJ/"
tmp_e1 = nc.Dataset(diri+"sys5_e1.nc","r")
tmp_e2 = nc.Dataset(diri+"sys5_e2.nc","r")
tmp_e3 = nc.Dataset(diri+"sys5_e3.nc","r")
tmp_e4 = nc.Dataset(diri+"sys5_e4.nc","r")
tmp_e5 = nc.Dataset(diri+"sys5_e5.nc","r")
tmp_e6 = nc.Dataset(diri+"sys5_e6.nc","r")
tmp_e7 = nc.Dataset(diri+"sys5_e7.nc","r")
tmp_e8 = nc.Dataset(diri+"sys5_e8.nc","r")
tmp_e9 = nc.Dataset(diri+"sys5_e9.nc","r")
tmp_e10 = nc.Dataset(diri+"sys5_e10.nc","r")
tmp_e11 = nc.Dataset(diri+"sys5_e11.nc","r")
tmp_e12 = nc.Dataset(diri+"sys5_e12.nc","r")
tmp_e13 = nc.Dataset(diri+"sys5_e13.nc","r")
tmp_e14 = nc.Dataset(diri+"sys5_e14.nc","r")
tmp_e15 = nc.Dataset(diri+"sys5_e15.nc","r")
tmp_e16 = nc.Dataset(diri+"sys5_e16.nc","r")
tmp_e17 = nc.Dataset(diri+"sys5_e17.nc","r")
tmp_e18 = nc.Dataset(diri+"sys5_e18.nc","r")
tmp_e19 = nc.Dataset(diri+"sys5_e19.nc","r")
tmp_e20 = nc.Dataset(diri+"sys5_e20.nc","r")
tmp_e21 = nc.Dataset(diri+"sys5_e21.nc","r")
tmp_e22 = nc.Dataset(diri+"sys5_e22.nc","r")
tmp_e23 = nc.Dataset(diri+"sys5_e23.nc","r")
tmp_e24 = nc.Dataset(diri+"sys5_e24.nc","r")
tmp_e25 = nc.Dataset(diri+"sys5_e25.nc","r")

obs_era =nc.Dataset(diri2+"filename","r")
obs_cru =nc.Dataset(diri3+"filename","r")
obs_merra2 =nc.Dataset(diri4+"filename","r")
obs_ncep =nc.Dataset(diri5+"filename","r")
obs_cpc =nc.Dataset(diri6+"filename","r")
obs_UoD = nc.Dataset(diri7+"filename","r")

#print(tmp_file.variables)
#may   = tmp_file.variables['m2']
#jun    = tmp_file.variables['m2']
lat     = tmp_e1.variables['lat'][:]
lon     = tmp_e1.variables['lon'][:]
time    = tmp_e1.variables['time'][:]


era=obs_era.variables['t2m'][:,:,:]
cru=obs_cru.variables['tmp'][:,:,:]
merra2=obs_merra2.variables['t2m'][:,:,:]
ncep=obs_ncep.variables['air'][:,:,:]
cpc=obs_cpc.variables['tmax'][:,:,:]
UoD=obs_UoD.variables['air'][:,:,:]

eramean=np.mean(era,axis=0)
crumean=np.mean(cru,axis=0)
merra2mean=np.mean(merra2,axis=0)
ncepmean=np.mean(ncep,axis=0)
cpcmean=np.mean(cpc,axis=0)
UoDmean=np.mean(UoD,axis=0)

nt, nlat, nlon = era.shape

nt2, nlat2, nlon2 = cru.shape
nt3, nlat3, nlon3 = UoD.shape

ngrd = nlon*nlat
# print(nt)
# print(ngrd)

#gobsmean=np.mean(gobsmay,axis=0)
eraanom=[]
eraanom=era-eramean


cruanom=[]
cruanom=cru-crumean

merra2anom=[]
merra2anom=merra2-merra2mean

ncepanom=[]
ncepanom=ncep-ncepmean

cpcanom=[]
cpcanom=cpc-cpcmean

UoDanom=[]
UoDanom=UoD-UoDmean

#print(lat.shape,lon.shape,time.shape)
#quit()

maye1 = tmp_e1.variables['m1'][:,:,:]
june1 = tmp_e1.variables['m2'][:,:,:]

msste1=np.mean((maye1+june1)/2,axis=0)
#print(msst.shape)
anom1=[]
anom1=(maye1+june1)/2-msste1

#print(anom2)
maye2 = tmp_e2.variables['m1'][:,:,:]
june2 = tmp_e2.variables['m2'][:,:,:]

msste2=np.mean((maye2+june2)/2,axis=0)
#print(msst.shape)
anom2=[]
anom2=(maye2+june2)/2-msste2

maye3 = tmp_e3.variables['m1'][:,:,:]
june3 = tmp_e3.variables['m2'][:,:,:]

msste3=np.mean((maye3+june3)/2,axis=0)
#print(msst.shape)
anom3=[]
anom3=(maye3+june3)/2-msste3


maye4 = tmp_e4.variables['m1'][:,:,:]
june4 = tmp_e4.variables['m2'][:,:,:]

msste4=np.mean((maye4+june4)/2,axis=0)
#print(msst.shape)
anom4=[]
anom4=(maye4+june4)/2-msste4


maye5 = tmp_e5.variables['m1'][:,:,:]
june5 = tmp_e5.variables['m2'][:,:,:]

msste5=np.mean((maye5+june5)/2,axis=0)
#print(msst.shape)
anom5=[]
anom5=(maye5+june5)/2-msste5


maye6 = tmp_e6.variables['m1'][:,:,:]
june6 = tmp_e6.variables['m2'][:,:,:]

msste6=np.mean((maye6+june6)/2,axis=0)
#print(msst.shape)
anom6=[]
anom6=(maye6+june6)/2-msste6

maye7 = tmp_e7.variables['m1'][:,:,:]
june7 = tmp_e7.variables['m2'][:,:,:]

msste7=np.mean((maye7+june7)/2,axis=0)
#print(msst.shape)
anom7=[]
anom7=(maye7+june7)/2-msste7

maye8 = tmp_e8.variables['m1'][:,:,:]
june8 = tmp_e8.variables['m2'][:,:,:]

msste8=np.mean((maye8+june8)/2,axis=0)
#print(msst.shape)
anom8=[]
anom8=(maye8+june8)/2-msste8


maye9 = tmp_e9.variables['m1'][:,:,:]
june9 = tmp_e9.variables['m2'][:,:,:]

msste9=np.mean((maye9+june9)/2,axis=0)
#print(msst.shape)
anom9=[]
anom9=(maye9+june9)/2-msste9

maye10 = tmp_e10.variables['m1'][:,:,:]
june10 = tmp_e10.variables['m2'][:,:,:]

msste10=np.mean((maye10+june10)/2,axis=0)
#print(msst.shape)
anom10=[]
anom10=(maye10+june10)/2-msste10


maye11 = tmp_e11.variables['m1'][:,:,:]
june11 = tmp_e11.variables['m2'][:,:,:]

msste11=np.mean((maye11+june11)/2,axis=0)
#print(msst.shape)
anom11=[]
anom11=(maye11+june11)/2-msste11
maye12 = tmp_e12.variables['m1'][:,:,:]
june12 = tmp_e12.variables['m2'][:,:,:]

msste12=np.mean((maye12+june12)/2,axis=0)
#print(msst.shape)
anom12=[]
anom12=(maye12+june12)/2-msste12

maye13 = tmp_e13.variables['m1'][:,:,:]
june13 = tmp_e13.variables['m2'][:,:,:]

msste13=np.mean((maye13+june13)/2,axis=0)
#print(msst.shape)
anom13=[]
anom13=(maye13+june13)/2-msste13

maye14 = tmp_e14.variables['m1'][:,:,:]
june14 = tmp_e14.variables['m2'][:,:,:]

msste14=np.mean((maye14+june14)/2,axis=0)
#print(msst.shape)
anom14=[]
anom14=(maye14+june14)/2-msste14

maye15 = tmp_e15.variables['m1'][:,:,:]
june15 = tmp_e15.variables['m2'][:,:,:]

msste15=np.mean((maye15+june15)/2,axis=0)
#print(msst.shape)
anom15=[]
anom15=(maye15+june15)/2-msste15


maye16 = tmp_e16.variables['m1'][:,:,:]
june16 = tmp_e16.variables['m2'][:,:,:]

msste16=np.mean((maye16+june16)/2,axis=0)
#print(msst.shape)
anom16=[]
anom16=(maye16+june16)/2-msste16


maye17 = tmp_e17.variables['m1'][:,:,:]
june17 = tmp_e17.variables['m2'][:,:,:]

msste17=np.mean((maye17+june17)/2,axis=0)
#print(msst.shape)
anom17=[]
anom17=(maye17+june17)/2-msste17

maye18 = tmp_e18.variables['m1'][:,:,:]
june18 = tmp_e18.variables['m2'][:,:,:]

msste18=np.mean((maye18+june18)/2,axis=0)
#print(msst.shape)
anom18=[]
anom18=(maye18+june18)/2-msste18


maye19 = tmp_e19.variables['m1'][:,:,:]
june19 = tmp_e19.variables['m2'][:,:,:]

msste19=np.mean((maye19+june19)/2,axis=0)
#print(msst.shape)
anom19=[]
anom19=(maye19+june19)/2-msste19


maye20 = tmp_e20.variables['m1'][:,:,:]
june20 = tmp_e20.variables['m2'][:,:,:]

msste20=np.mean((maye20+june20)/2,axis=0)
#print(msst.shape)
anom20=[]
anom20=(maye20+june20)/2-msste20


maye21 = tmp_e21.variables['m1'][:,:,:]
june21 = tmp_e21.variables['m2'][:,:,:]

msste21=np.mean((maye21+june21)/2,axis=0)
#print(msst.shape)
anom21=[]
anom21=(maye21+june21)/2-msste21

maye22 = tmp_e22.variables['m1'][:,:,:]
june22 = tmp_e22.variables['m2'][:,:,:]

msste22=np.mean((maye22+june22)/2,axis=0)
#print(msst.shape)
anom22=[]
anom22=(maye22+june22)/2-msste22

maye23 = tmp_e23.variables['m1'][:,:,:]
june23 = tmp_e23.variables['m2'][:,:,:]

msste23=np.mean((maye23+june23)/2,axis=0)
#print(msst.shape)
anom23=[]
anom23=(maye23+june23)/2-msste23
maye24 = tmp_e24.variables['m1'][:,:,:]
june24 = tmp_e24.variables['m2'][:,:,:]

msste24=np.mean((maye24+june24)/2,axis=0)
#print(msst.shape)
anom24=[]
anom24=(maye24+june24)/2-msste24

maye25 = tmp_e25.variables['m1'][:,:,:]
june25 = tmp_e25.variables['m2'][:,:,:]

msste25=np.mean((maye25+june25)/2,axis=0)
#print(msst.shape)
anom25=[]
anom25=(maye25+june25)/2-msste2


lonE=65
lonW=80
latS=31
latN=39
lon1=np.where(lon==lonE)
lon2=np.where(lon==lonW)
lat1=np.where(lat==latS)
lat2=np.where(lat==latN)
#print(lon1,lon2,lat1,lat2)
lonx, latx = np.meshgrid(lon, lat)
weights = np.cos(latx * np.pi / 180.)
ts_era = np.zeros(nt)
ts_cru = np.zeros(nt)
ts_merra2 = np.zeros(nt)
ts_ncep = np.zeros(nt)
ts_cpc = np.zeros(nt)
ts_UoD = np.zeros(nt)



ts_sati1 = np.zeros(nt)
ts_sati2 = np.zeros(nt)
ts_sati3 = np.zeros(nt)
ts_sati4 = np.zeros(nt)
ts_sati5 = np.zeros(nt)

ts_sati6 = np.zeros(nt)
ts_sati7 = np.zeros(nt)
ts_sati8 = np.zeros(nt)
ts_sati9 = np.zeros(nt)
ts_sati10 = np.zeros(nt)

ts_sati11 = np.zeros(nt)
ts_sati12 = np.zeros(nt)
ts_sati13 = np.zeros(nt)
ts_sati14 = np.zeros(nt)
ts_sati15 = np.zeros(nt)

ts_sati16 = np.zeros(nt)
ts_sati17 = np.zeros(nt)
ts_sati18 = np.zeros(nt)
ts_sati19 = np.zeros(nt)
ts_sati20 = np.zeros(nt)

ts_sati21 = np.zeros(nt)
ts_sati22 = np.zeros(nt)
ts_sati23 = np.zeros(nt)
ts_sati24 = np.zeros(nt)
ts_sati25 = np.zeros(nt)

    
#ts_sati=[]
for it3 in np.arange(nt3):
    ts_UoD[it3] = np.ma.average(UoDanom[it3, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])

for it2 in np.arange(nt2):
    ts_cru[it2] = np.ma.average(cruanom[it2, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])

for it in np.arange(nt):
    
    ts_era[it] = np.ma.average(eraanom[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_ncep[it] = np.ma.average(ncepanom[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_merra2[it] = np.ma.average(merra2anom[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_cpc[it] = np.ma.average(cpcanom[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    
    
    
    ts_sati1[it] = np.ma.average(anom1[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati2[it] = np.ma.average(anom2[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati3[it] = np.ma.average(anom3[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati4[it] = np.ma.average(anom4[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati5[it] = np.ma.average(anom5[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    
    ts_sati6[it] = np.ma.average(anom6[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati7[it] = np.ma.average(anom7[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati8[it] = np.ma.average(anom8[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati9[it] = np.ma.average(anom9[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati10[it] = np.ma.average(anom10[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    
    ts_sati11[it] = np.ma.average(anom11[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati12[it] = np.ma.average(anom12[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati13[it] = np.ma.average(anom13[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati14[it] = np.ma.average(anom14[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati15[it] = np.ma.average(anom15[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    
    ts_sati16[it] = np.ma.average(anom16[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati17[it] = np.ma.average(anom17[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati18[it] = np.ma.average(anom18[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati19[it] = np.ma.average(anom19[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati20[it] = np.ma.average(anom20[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    
    ts_sati21[it] = np.ma.average(anom21[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati22[it] = np.ma.average(anom22[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati23[it] = np.ma.average(anom23[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati24[it] = np.ma.average(anom24[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])
    ts_sati25[it] = np.ma.average(anom25[it, int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])], weights=weights[int(lat1[0]):int(lat2[0]),int(lon1[0]):int(lon2[0])])

ts_mean=ts_sati1+ts_sati2+ts_sati3+ts_sati4+ts_sati5+ts_sati6+ts_sati7+ts_sati8+ts_sati9+ts_sati10+ts_sati11+ts_sati12+ts_sati13+ts_sati14+ts_sati15+ts_sati16+ts_sati17+ts_sati18+ts_sati19+ts_sati20+ts_sati21+ts_sati22+ts_sati23+ts_sati24+ts_sati25
    

ts_meanen=ts_mean/25
print(ts_meanen.shape)
ts_satisd=np.std(ts_meanen)        

std_meanen = np.std(ts_meanen)
error_meanen = std_meanen/np.sqrt(len(ts_meanen))    
    
   
fig,ax =plt.subplots()

corr_era, _ = pearsonr(ts_meanen, ts_era)
print('Pearsons correlation (ERA5): %.2f' % corr_era)
label_era = "{:.2f}".format(corr_era)

std_era = np.std(ts_era)
error_era = std_era/np.sqrt(len(ts_era))
#error_era = std_era/len(ts_era)

print(error_era)
corr_ncep, _ = pearsonr(ts_meanen, ts_ncep)
print('Pearsons correlation (NCEP): %.2f' % corr_ncep)
label_ncep = "{:.2f}".format(corr_ncep)
std_ncep = np.std(ts_ncep)
error_ncep = std_ncep/np.sqrt(len(ts_ncep))
print(error_ncep)

corr_merra2, _ = pearsonr(ts_meanen, ts_merra2)
print('Pearsons correlation (MERRA2): %.2f' % corr_merra2)
label_merra2 = "{:.2f}".format(corr_merra2)

std_merra2 = np.std(ts_merra2)
error_merra2 = std_merra2/np.sqrt(len(ts_merra2))
print(error_merra2)
corr_cru, _ = pearsonr(ts_meanen, ts_cru)
print('Pearsons correlation (CRU): %.2f' % corr_cru)
label_cru = "{:.2f}".format(corr_cru)

std_cru = np.std(ts_cru)
error_cru = std_cru/np.sqrt(len(ts_cru))
print(error_cru)
corr_cpc, _ = pearsonr(ts_meanen, ts_cpc)
print('Pearsons correlation (CPC): %.2f' % corr_cpc)
label_cpc = "{:.2f}".format(corr_cpc)

std_cpc = np.std(ts_cpc)
error_cpc = std_cpc/np.sqrt(len(ts_cpc))
print(error_cpc)

plt.ylim(ymin=-4.5, ymax=4.5)
plt.xlim(1980,2023,2)
#plt.plot(np.arange(1981,2023,1),ts_sati1,marker='x',color='blue',label='CMC')
plt.scatter(np.arange(1981,2023,1),ts_sati1,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati2,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati3,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati4,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati5,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati6,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati7,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati8,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati9,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati10,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati11,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati12,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati13,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati14,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati15,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati16,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati17,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati18,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati19,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati20,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati21,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati22,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati23,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati24,marker='o',facecolors='none',color='grey')
plt.scatter(np.arange(1981,2023,1),ts_sati25,marker='o',facecolors='none',color='grey',label='SEAS5-mem')
#plt.plot(np.arange(1981,2023,1),ts_UoD,color='olive',linestyle='dashed',label='UoD (%s)'%label11 )
plt.plot(np.arange(1981,2023,1),ts_meanen,marker='x',color='blue',label='SEAS5-Ensmean')

plt.plot(np.arange(1981, 2023, 1), ts_era, color='green', label='ERA5 (%s±%.2f)' % (label_era, error_era))
#plt.errorbar(np.arange(1981, 2023, 1), ts_era, yerr=error_era, fmt='o', markersize=4, color='green', ecolor='green', capsize=2)

plt.plot(np.arange(1981,2023,1),ts_ncep,color='black',linestyle='dashed',label='NCEP (%s±%.2f)' % (label_ncep, error_ncep))
#plt.errorbar(np.arange(1981, 2023, 1), ts_ncep, yerr=error_ncep, fmt='o', markersize=4, color='black', ecolor='black', capsize=2)

plt.plot(np.arange(1981,2023,1),ts_merra2,color='orange',linestyle='dashed',label='MERRA2 (%s±%.2f)' % (label_merra2, error_merra2))
#plt.errorbar(np.arange(1981, 2023, 1), ts_merra2, yerr=error_merra2, fmt='o', markersize=4, color='orange', ecolor='orange', capsize=2)

plt.plot(np.arange(1981,2023,1),ts_cru,color='magenta',linestyle='dashed',label='CRU (%s±%.2f)' % (label_cru, error_cru))
#plt.errorbar(np.arange(1981, 2023, 1), ts_cru, yerr=error_cru, fmt='o', markersize=4, color='magenta', ecolor='magenta', capsize=2)

plt.plot(np.arange(1981,2023,1),ts_cpc,color='cyan',linestyle='dashed',label='CPC (%s±%.2f)' % (label_cpc, error_cpc))
#plt.errorbar(np.arange(1981, 2023, 1), ts_cpc, yerr=error_cpc, fmt='o', markersize=4, color='cyan', ecolor='cyan', capsize=2)

ax.hlines(0,1980,2023, color='black', linewidth=0.5)
plt.ylim(-4,4)
#plt.xlim(-3,3)
# plt.xticks(fontsize=12,weight='bold')
# plt.yticks(fontsize=12,weight='bold')
plt.xlabel('Years', fontsize=12)
plt.ylabel('SAT anomaly(°C)', fontsize=12)
#plt.legemj(hamjles=[p1,p2], loc='upper left')
ax.legend(prop={'size': 8}, fontsize='medium',loc='lower left',ncol=3)
plt.title('Prediction skill of the SATI',fontsize=14,loc = "left")
plt.savefig('skill-ALL-DATAvsMOD.png', dpi = 300)
#plt.show()
#quit()
###########################################################


