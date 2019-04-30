
from isrp import *

sensors = isrpLoadSensorParameters('rogerpass','/home/sandro/Documents/isrpDtCalculator')

demFilename ='/home/sandro/Documents/isrpDtCalculator/dem/rogerpass/rogerpass-ritaglio.asc'
sRes=150
sRes1=170
dMin=0
dMax=20000
fig = plt.figure(num=1,figsize=(12, 8))
#
# ax1 = fig.add_subplot(121)
xdem, ydem, zdem = isrpLoadDem(demFilename,sensors,4000)
# plt.contourf(xdem,ydem, zdem, cmap="gray",levels=list(range(0, 5000, 60)))
# plt.title("Elevation Contours ")
# cbar = plt.colorbar(orientation="horizontal")
# plt.gca().set_aspect('equal', adjustable='box')
# lineMap, = plt.plot(np.nan,np.nan, linestyle='dashed', linewidth=1, color='b')
# ax1.plot(sensors['X'][7], sensors['Y'][7], marker='*', color='r', markersize=12)
# ax1.plot(sensors['X'][0], sensors['Y'][0], marker='*', color='b', markersize=12)
# ax2 = fig.add_subplot(122)
# lineProfile, = plt.plot(0, 0, linewidth=1, color='b')
# lineSound, = plt.plot(0, 0,marker='o', markersize=6, linestyle='dashed', linewidth=1, color='r')
# #ax2.set_xlim([0,20000])
# ax2.set_ylim([1000,5000])
# plt.grid(color='k', linestyle='-', linewidth=.1)
# plt.draw()
# plt.pause(0.1)

#dT=isrpDemTravelDt(demFilename,xdem,ydem,zdem,sensors,sRes,dMin,dMax)
loadT=np.load(demFilename+'dT.npz')
dT=loadT['dT']
T=loadT['T']


corrM=np.array([
               [1, 1, 1, 1, 0, 0, 0, 0,0],
               [1, 1, 1, 1, 0, 0, 0, 0,0],
               [1, 1, 1, 1, 0, 0, 0, 0,0],
               [1, 1, 1, 1, 0, 0, 0, 0,0],
               [0, 0, 0, 0, 1, 1, 1, 1,1],
               [0, 0, 0, 0, 1, 1, 1, 1,1],
               [0, 0, 0, 0, 1, 1, 1, 1,1],
               [0, 0, 0, 0, 1, 1, 1, 1,1],
               [0, 0, 0, 0, 1, 1, 1, 1, 1]
                ])

dMap=isrpArrange2(demFilename,-dT,T,sensors,sRes1,1700,50/330)

fig = plt.figure(num=2,figsize=(12, 8))
aa=np.float64(dT[1,2,:,:].T)
plt.subplot(2,2,1)
plt.imshow(aa)
aa=np.float64(dT[3,2,:,:].T)
plt.subplot(2,2,2)
plt.imshow(aa)
aa=np.float64(dT[7,6,:,:].T)
plt.subplot(2,2,3)
plt.imshow(aa)
aa=np.float64(dT[7,8,:,:].T)
plt.subplot(2,2,4)
plt.imshow(aa)
plt.colorbar()
plt.draw()
plt.show()
