import os
import json
import numpy as np
from numpy import hstack
from numpy import vstack
from scipy import signal
from scipy.signal import butter, lfilter
import scipy.io as sio
import scipy.ndimage

import urllib.request
from urllib.parse import unquote
import datetime as dt

import math
from math import cos, pi, sqrt, floor

import utm

import imp

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import glob
import fnmatch

import requests
import wget

from collections import namedtuple

from joblib import Parallel, delayed
import multiprocessing


sensorsType = np.dtype({'names': ('name', 'configFilename', 'X', 'Y', 'Z'),
                          'formats': ('U10', 'U60', 'f8', 'f8', 'f8')})


def isrpLoadSensorParameters(network,path):
    modname = 'stationparameters'
    path_config =path+'/configfiles/'+network+'/'
    listOfFiles = os.listdir(path_config)
    pattern = "*.txt"

    i = 0
    file_config = {}
    for entry in listOfFiles:
        if fnmatch.fnmatch(entry, pattern):
            file_config[i] = path_config + entry
            i += 1

    nSensors=len(file_config)
    sensors=np.zeros(nSensors, dtype=sensorsType)
    i = 0
    for entry in file_config:
        print(file_config[entry])
        I = imp.load_source(modname, file_config[entry])
        sensors['name'][i] = I.id
        sensors['configFilename'] = file_config[entry]
        sensors['X'][i] = I.x
        sensors['Y'][i] = I.y
        sensors['Z'][i] = I.z
        i += 1
    return sensors


def isrpLoadDem(demFilename,sensors,iRange):
    f = open(demFilename, 'r')
    whl = f.read()
    whl = whl.split('\n')
    # header
    hdr = whl[0:5]
    deminfo = np.dtype({'names': ('ncols', 'nrows', 'xllcorner', 'yllcorner', 'dx', 'dy'),
                        'formats': ('f8', 'f8', 'f8', 'f8', 'f8', 'f8')})
    nInfo = len(hdr)
    info = np.zeros(1, dtype=deminfo)
    s = hdr[0].split()
    info['ncols'] = float(s[1])-1
    s = hdr[1].split()
    info['nrows'] = float(s[1])
    s = hdr[2].split()
    info['xllcorner'] = float(s[1])
    s = hdr[3].split()
    info['yllcorner'] = float(s[1])
    s = hdr[4].split()
    info['dx'] = float(s[1])
  #  s = hdr[5].split()
    info['dy'] = float(s[1])

    xllcu, yllcu, zonen, zonel = utm.from_latlon(info['yllcorner'], info['xllcorner'])
    print('utm zone: ' + str(zonen) + zonel)

    # Latitude:  1 deg = 110.54 km
    # Longitude: 1 deg = 111.320*cos(latitude) km
    dy = info['dy'] * 110.54 * 1000
    dx = info['dx'] * 111.32 * cos(info['yllcorner'] * pi / 180) * 1000

    xu = np.arange(xllcu, xllcu + info['ncols'] * dx, dx)
    #print(info['ncols'])
    yu = np.arange(yllcu + info['nrows'] * dy, yllcu, -dy)
    #print(info['nrows'])

    # elevation matrix
    bb = whl[5:len(whl)]
    print(len(bb))
    idx = 0
    for i in range(0, len(yu)):
       # print(i);
        bw = bb[i].split(' ')
        bw.remove('')
        aa = np.asarray(bw, dtype=np.float32)
        if idx == 0:
            z = aa
        else:
            z = vstack((z, aa))
        idx += 1





    print('ncols: ', str(len(xu)))
    print('nrows: ', str(len(yu)))
    print('zmatrix: ' + z.shape[0].__str__() + ' x ' + z.shape[1].__str__())

    # int(np.amax(z))
    #plt.contourf(xu, yu, z, cmap="gray",
     #            levels=list(range(0, 5000, 50)))
    #plt.title("Elevation Contours Goms (CH) area")
    #cbar = plt.colorbar()
    #plt.gca().set_aspect('equal', adjustable='box')
    # lat=5146098.00 m N
    # lon=442227.00 m E
    print(len(sensors))
    #for i in range(0,len(sensors)):
    #    print(i)
    #    plt.plot(sensors['X'][i], sensors['Y'][i], marker='o', markersize=2, color='r', ls='')

    #plt.show()
    z[np.isnan(z)] = 0

    xL = np.where(xu > (np.min(sensors['X']) - iRange))[0][0]
    xR = np.where(xu > (np.max(sensors['X']) + iRange))[0][0]
    yB = np.where(yu < (np.max(sensors['Y']) - iRange))[0][0]
    yT = np.where(yu < (np.min(sensors['Y']) + iRange))[0][0]

    xu = xu[xL:xR]
    yu = yu[yT:yB]
    z = z[yT:yB, xL:xR]

    return xu, yu, z


def isrpTravelTimesComput(x0, y0, xS, yS, x1, y1, xdem ,ydem ,zdem,sRes,dMin,dMax):
    # x0 = 13  node xposition index
    # x1 = 255 sensor xposition index
    # y0 = 135 node yposition index
    # y1 = 222 sensor yposition index
    # demres resolution
    #fig2 = plt.figure(num=1,figsize=(12, 8))



    dPlane = (sqrt((xS - xdem[x0]) ** 2 + (yS - ydem[y0]) ** 2))

    if((dPlane>dMin)and(dPlane<dMax)):
        num = max(floor(dPlane/sRes),2)

        xp, yp = np.linspace(x0, x1, num), np.linspace(y0, y1, num)
        zp = scipy.ndimage.map_coordinates(zdem, np.vstack((yp, xp)))
        dr = dPlane/(len(zp)-1)
        # (len(zp) - 1)
        cr = dr * np.arange(0, len(zp), dtype=np.int)
        nd = len(cr)

        mm = (zp[-1] - zp[0]) / cr[-1]  # diff quota / distanza (coeff ang o elevation)
        r = mm * cr + zp[0]  # retta con coeff. ang pari a elevation tra array e nodo
        r.transpose()
        h = np.fix((zp - r) / 10) * 10  # 10 = z resolution (m)
        h[np.nonzero(h < 0)] = 0
        h.transpose()

        k = 0
        io = 0
        ii=[]
        xx=[]
        yy=[]

        while io < nd - 1:
            for i in range(io, nd):

                m1 = (h[i] - h[io]) / (cr[i] - cr[io])
                n1 = (h[i] * cr[io] - h[io] * cr[i]) / (cr[io] - cr[i])
                r1 = m1 * cr[io:i] + n1
                h1 = np.fix((h[io:i] - r1) / 10) * 10
                ih = h1 > 0
                ih = ih.astype(np.int)
                if np.sum(ih) == 0:
                    ik = i

                ii.append(ik)
                xx.append(cr[io])
                yy.append(zp[io])
                k += 1

            io = ik

        xx = np.append(xx, cr[-1])
        yy = np.append(yy, zp[-1])

        dTopo = 0
        for i in range(1, k + 1):
            dTopo = dTopo + sqrt((xx[i] - xx[i - 1]) ** 2 + (yy[i] - yy[i - 1]) ** 2)
        tT = dTopo

        # lineMap.set_xdata([xu[x0] , xS])
        # lineMap.set_ydata([yu[y0] , yS])
        #
        # lineProfile.set_xdata(cr)
        # lineProfile.set_ydata(zp)
        #
        # lineSound.set_xdata(xx)
        # lineSound.set_ydata(yy)
        # plt.sca(ax2)
        # plt.gca().relim()
        # plt.gca().autoscale_view()
        # plt.pause(0.05)

    else:
        tT=-1

    # print("dPlane"+str(dPlane))
    #print("dtopo "+str(dTopo))
    return tT


def isrpParallelDemTravelDt(i,xdem, ydem, zdem, sensors:sensorsType,sRes,dMin,dMax):
    tT = np.zeros(( len(xdem), len(ydem)),dtype=np.float16)

    dx = xdem[1] - xdem[0]
    dy = ydem[1] - ydem[0]

    x1 = np.where(xdem > sensors['X'][i])
    y1 = np.where(ydem < sensors['Y'][i])

    x1 = x1[0][0]
    y1 = y1[0][0]
    xS = sensors['X'][i]
    yS = sensors['Y'][i]

    dxS = (xdem[x1] - xS) / dx
    dyS = (ydem[y1] - yS) / dy

    x1 = x1 + dxS
    y1 = y1 + dyS
    for xi in range(0,len(xdem)):
        print("s " +str(i) +"xi "+str(xi))
        for yi in range(0,len(ydem)):
            if (zdem[yi, xi]>0):
                tT[xi, yi] = isrpTravelTimesComput(xi, yi,xS, yS, x1, y1,xdem,ydem,zdem,sRes,dMin,dMax)
            else :
                tT[xi, yi] = -1
    return tT


def isrpDemTravelDt(demFilename, xdem, ydem, zdem, sensors:sensorsType,sRes, dMin,dMax):

    dT=np.zeros((len(sensors), len(sensors), len(xdem), len(ydem)),dtype=np.float16)
    T = np.zeros((len(sensors), len(xdem), len(ydem)),dtype=np.float16)

    num_cores = multiprocessing.cpu_count()

    T[:]=Parallel(n_jobs=num_cores-1)(delayed(isrpParallelDemTravelDt)(i,xdem, ydem, zdem, sensors,sRes,dMin,dMax) for i in range(0,len(sensors)))

    for i in range(0,len(sensors)):
        for ii in range(i+1, len(sensors)):
            print( "isrpDemTravelDt dElaborating sernsor couple "+str(i)+" "+str(ii))
            for xi in range(0,len(xdem)):
                for yi in range(0,len(ydem)):
                    if (T[ii,xi,yi]>0):
                        dT[i, ii, xi, yi] = T[ii,xi,yi]-T[i,xi,yi]
                    else :
                        dT[i, ii, xi, yi] = -1
    np.savez(demFilename+'dT',dT=dT,T=T,dMin=dMin,dMax=dMax)
    return dT


def isrpArrange (demFileName,dT,T,sensors,shift,sToSFactor):
    dS = dT * sToSFactor
    S = T * sToSFactor
    sShift=shift*sToSFactor
    dMap=[]
    sMap=[]
    dsMin = np.nanmin(dS)
    dsMax = np.nanmax(dS)

    for i in range(0,len(sensors)):
        dii=[]
        for ii in range(0, len(sensors)):
            print("isprArrange dElaborating sernsor couple " + str(i) + " " + str(ii))
            dj=[]
            for j in range(int(dsMin-1),int(dsMax+1)):
              #  print("isprArrange Elaborating T " + str(j))
                dj.append(np.where((dS[i,ii,:,:] >= j) & (dS[i,ii,:,:]< j+1)))
            dii.append(dj)
        dMap.append(dii)

    for i in range(0, len(sensors)):
        dj = []
        for j in range(0, int(np.nanmax(S)),int(np.nanmax(S)/sShift)):
            print("isprArrange Elaborating T " + str(j))
            dj.append(np.where((S[i, :, :] >= j) & (S[i, :, :] < j + sShift)))
        sMap.append(dj)


    np.savez(demFileName + 'dMap', dMap=dMap,sMap=sMap,minCorr=dsMin,maxCorr=dsMax)
    return dMap, sMap


def isrpGetWyData(sensors,ti_str,tf_str):
    #tmin.strftime('%Y-%m-%d%%20%H:%M:00'), tmax.strftime('%Y-%m-%d%%20%H:%M:%S')
    idx = 0
    for i in sensors:
        try:
            I = imp.load_source('stationparameters', i['configFilename'])
            domain = I.wydomain
            location = I.wylocation
            key = I.key
            station = I.id
            args = "key={key}&id={id}&json=true&tmin={tmin}&tmax={tmax}".format(id=station, tmin=tmin, tmax=tmax,
                                                                                key=key)
            req = "{d}/{p}?{args}".format(d=domain, p=location, args=args)

            # Load the data
            with urllib.request.urlopen(req) as url:
                data = json.loads(url.read().decode())
            out = np.array(data["values"])
            a = out[:, 1]

        except:
            print("    > channel " + i + " failed")
            continue

        if idx == 0:
            data = a
        else:
            data = vstack((data, a))

        idx += 1
        #out.tt = datenum(1970, 1, 1) + time'/86400;

        if len(data) == 0:
            print('... WARNING: NO DATA retrieved from WWS >>> quitting !')
            quit()

    secs = out[:, 2]
    timestamp = [datetime.date(1970, 1, 1)+datetime.timedelta(seconds=i) for i in secs]

    return data, timestamp



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

