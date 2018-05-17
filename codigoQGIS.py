import ee
import matplotlib.pylab as plt
import numpy as np
import matplotlib.cm as cm
import time
import json
from datetime import datetime, timedelta
import time
from datetime import date

ee.Initialize()
start_time = time.time()
layer = qgis.utils.iface.activeLayer()
layer.name()
selected_features = layer.selectedFeatures()
count= len(selected_features)
print("selected features", count)
landsatCollection = ee.ImageCollection('COPERNICUS/S2');
landsatDateCollection = ee.ImageCollection(landsatCollection.filterDate('2017-05-01', '2017-05-31'))
print("Imagenes filtradas por fecha", landsatDateCollection.size().getInfo())
i=0
val = [None] * count
etiquetas = []

while i< count:
    p = selected_features[i]
    print("P", p.attributes()[1])
    etiquetas.append( p.attributes()[1])
    print("ETIQUETAS", etiquetas)
    g = p.geometry()
    gJSON=g.exportToGeoJSON()
    o=json.loads(gJSON)
    print("GEOMETRY 2", o)
    landsatAOI = ee.ImageCollection(landsatDateCollection.filterBounds(o))
    print("STOP 1", landsatAOI.size().getInfo())
    images = [item.get('id') for item in landsatAOI.getInfo().get('features')]
    #print("STOP 2", len(images))
    feature = ee.Feature(o)
    print("FEATURE", feature.geometry().getInfo())
    values=[]
    for image in images:
        print("IMAGE",ee.Image(image).reduceRegion(ee.Reducer.mean(), feature.geometry()).getInfo())
        nir = ee.Image(image).reduceRegion(ee.Reducer.mean(), feature.geometry()).getInfo().get('B8')
        red = ee.Image(image).reduceRegion(ee.Reducer.mean(), feature.geometry()).getInfo().get('B4')
        print("NIR", nir)
        print("RED", red)
        ndvi = ((nir - red)/(nir + red))
        print("NDVI", ndvi)
        values.append(ndvi)
    val[i]=values
    i+=1

cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0, 1, count)]
#date1= date(2017,05,01)
date_N_days= date(2017,05,01)
countimages=  landsatAOI.size().getInfo()
fechas=[None] * countimages
print(date_N_days)
for i in range (0,landsatAOI.size().getInfo()): #de 0 a 4
    fechas[i]= date_N_days
    date_N_days = date_N_days + timedelta(days=5)

print("fechas", fechas[0])
for i, color in enumerate(colors, start=0):
    plt.plot(fechas,val[i],color=color,linewidth=2)
plt.legend( etiquetas, loc = 'upper right')
plt.xlabel('Date')
plt.ylabel('NDVI value')
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))

#PRUEBA
#p = selected_features[4]
#atr=p.attributes()
#etiquetas.append(atr[1])
#print("ETIQUETAS", etiquetas)
#g = p.geometry()
#gJSON=g.exportToGeoJSON()
#print("GEOMETRY", gJSON)
#FIN PRUEBA