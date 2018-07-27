import ee
import matplotlib.pylab as plt
import matplotlib.pyplot as pl
import numpy as np
import matplotlib.cm as cm
import time
import json
import datetime
import copy
import shutil
import pandas as pd
from PyQt4.QtGui import QColor
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from scipy.stats.stats import pearsonr
from datetime import datetime, timedelta
from datetime import date
from sklearn.metrics import confusion_matrix

ee.Initialize()
start_time = time.time()

layer = qgis.utils.iface.activeLayer()
print("Nombre de layer", layer.name())

#features seleccionadas
selected_features = layer.selectedFeatures()
count= len(list(selected_features))
print("Feature number", count)
#Filtrar las fotos por fecha
sentinelCollection = ee.ImageCollection('COPERNICUS/S2');
sentinelDateCollection = ee.ImageCollection(sentinelCollection.filterDate('2017-06-21', '2017-09-23'))

i=0
val = [None] * count
valGNDVI = [None] * count
etiquetas = []
clases= []
datak = [None] * count
fechasRep = [None] * count
dife= [None] * count
datosBosque=[]
datosCesped=[]
hayBosque= False
hayCesped= False
featuresNull = []
featuresNotNull=[]

#analizar cada feature
while i< count:
    
    p = selected_features[i]
    #crear un vector con el nombre de cada feature. p[1] el el atributo de descripcion
    etiquetas.append(p[1])
#    if p[1]==NULL:
#        print 1
#        clases.append(str(p[3]))
        
    print("ETIQUETAS", etiquetas)
    g = p.geometry()
    gJSON=g.exportToGeoJSON()
    o=json.loads(gJSON)
    b = ee.Geometry.Polygon(o['coordinates'])
    #filtrar la coleccion de imagenes por la geometria de la feature que se esta analizando
    sentinelAOI = ee.ImageCollection(sentinelDateCollection.filterBounds(b))
    print("Imagenes filtradas por feature", sentinelAOI.size().getInfo())
    imagesConNubes = [item.get('id') for item in sentinelAOI.getInfo().get('features')]
    print("STOP 2", len(imagesConNubes))
    
    
    #crear dos arrays para separar las features sin nobre de las nombradas(mar, arena, bosque...)
    if p[1]==NULL:
        print("DENTRO NULL")
        featuresNull.append(p)
    else:
        featuresNotNull.append(p)
    values=[]
    valuesGNDVI=[]
    datas=[]
    diferencias=[]
    fechasR = []
    images=[]
    #crear coleccion de imagenes sin nubes
    for icn in imagesConNubes:
        icnindex= imagesConNubes.index(icn)
        cloudPercentage=ee.Image(imagesConNubes[icnindex]).get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        if cloudPercentage< 10.0:
            images.append(icn)
        else:
            print("IMAGEN CON MUCHAS NUBES")
    print("IMAGENES SIN NUBES", len(images))
    #analizar cada imagen de la coleccion
    for image in images:
        print 1
        #conseguir la fecha de recepcion de la imagen
        date1 = ee.Image(image).get('system:time_start')
        indexPhoto= images.index(image)
        dat= date.fromtimestamp(date1.getInfo() / 1e3)
        print("FECHA", dat)
        datas.append(dat)

            
        #properties = ee.Image(image).propertyNames();
        
        # calcular NDVI
        nir = ee.Image(image).reduceRegion(ee.Reducer.mean(), o).getInfo().get('B8')
        red = ee.Image(image).reduceRegion(ee.Reducer.mean(), o).getInfo().get('B4')
        
        #en caso de que nir o red sean nulos, darle el mismo valor que el contrario para formzar que
        #NDVI sea 0 e imputar los valores posteriormente
        
        if nir == None:
            print("NIR IS NONE")
            nir = red
        if red == None:
            print("RED IS NONE")
            red = nir
                    
        if nir == 0 and red == 0:
            ndvi= 0
        else:
            ndvi = ((nir - red)/(nir + red))
        
        values.append(ndvi)
        
        #GNDVI
        green = ee.Image(image).reduceRegion(ee.Reducer.mean(), o).getInfo().get('B3')
        if green == None:
            print("GREEN IS NONE")
            green = nir
                    
        if nir == 0 and green == 0:
            ndvi= 0
        else:
            gndvi = ((nir - green)/(nir + green))
        
        valuesGNDVI.append(gndvi)
        
        etiquetasR= []
        
        #Cuando hay dos fotos de la misma fecha
        
        if indexPhoto>0 :

            a= copy.copy(datas[indexPhoto])
            b= copy.copy(datas[indexPhoto-1])

            if a == b: #si hay dos fechas iguales
                print("FECHA REPETIDA", a)
                fechasR.append(a)
                etiquetasR.append(p[1])
                if p[1]=='bosque':
                    hayBosque= True
                    print("Bosque index i e index i-1", values[indexPhoto],values[indexPhoto-1])
                    datosBosque.append(max([values[indexPhoto],values[indexPhoto-1]]))
                    print("Maximo bosque", max([values[indexPhoto],values[indexPhoto-1]]))
                elif p[1]== 'cesped':
                    hayCesped= True
                    print("cesped index i e index i-1", values[indexPhoto],values[indexPhoto-1])
                    datosCesped.append(max([values[indexPhoto],values[indexPhoto-1]]))
                    print("Maximo cesped", max([values[indexPhoto],values[indexPhoto-1]]))
                
                dif= abs(values[indexPhoto]-values[indexPhoto-1])
                diferencias.append(dif)
                

                
            
    #Todas las imagenes procesadas de una feature
    dife[i]=diferencias

    fechasRep[i]=fechasR
    if i == 0:
        tamanoColumnas= datas
    else:
        if len(tamanoColumnas)< len(datas):
            tamanoColumnas= datas
            print("Tamano columnas", tamanoColumnas)
    
    datak[i]=datas
    val[i]=values
    print ("FECHA Y DATOS", datak[i], val[i])
   
    valGNDVI[i]= valuesGNDVI
    
    
    
    # Grafica de la diferencia de valores para imagenes tomadas en la misma fecha
    
   
    h=plt.figure(1)
    plt.plot(fechasRep[i][:],dife[i][:],color='silver', linewidth=3, alpha=0.8)
    pr=[str(j) for j in fechasRep[i][:]]
    #x= range(len(fechasRep[i][:]))
    #fig, ax = plt.subplots(1,1)  
    #ax.set_xticklabels(pr, rotation='vertical', fontsize=10)
    #plt.xticks(range(len(pr)), pr, size='small')
    plt.xticks(fontsize=26)
    ax = h.add_subplot(111)
    ax.set_xlabel('Fechas repetidas', fontsize=30)
    ax.set_ylabel('Diferencia NDVI', fontsize=30)
    #plt.xlabel('Fechas repetidas')
    #plt.ylabel('Diferencia NDVI')
    
    
    i+=1
#Deberia de etsar seleccionado el feature de bosque y cesped para cuantificar la diferencia
if hayBosque and hayCesped:
    
    print("DATOS DATOSBOSQUE ", datosBosque )
    print("DATOS DATOSCESPED", datosCesped)

    difCB= [abs(x - y) for x, y in zip(datosBosque, datosCesped)]
    print("DIFERENCIAS BOSQUE Y CESPED",difCB)
    plt.plot(fechasRep[0][:],difCB,color='red',linewidth=3, alpha=0.8)
h.show()



#Crear fichero
print("DATA Y COLUMNAS", val[:], tamanoColumnas)

fp1= open('/tmp/datosTimeBased.txt', 'w')
fp2= open('/tmp/datosSuma.txt', 'w')
fp3= open('/tmp/datosPrueba.txt', 'w')

#crear un dataframe con los valores NDVI calculados, los index seran las etiquetas
#y las columnas seran las fechas.
fp= open('/tmp/datos.txt', 'w')
fechasColumnas=[str(i) for i in tamanoColumnas]
df = pd.DataFrame(data=val[:],columns=fechasColumnas)
df['etiqueta'] = etiquetas


print("REPLACE")
#buscar los valores 0 y sustiturilos por n.a para hacer interpolacion linear
dfn= df.replace(0, np.nan)
dfni=dfn.interpolate()
numrows=len(dfni.index)
#dfni= dfni.sort_index(axis='index',na_position='first')
#indexlevels=dfni.index.values
dfni=dfni.sort_values(by=['etiqueta'], na_position='first')
dfnicopia=dfni.copy()
nulls=len(featuresNull)
nonulls=len(featuresNotNull)
fp7= open('/tmp/dfnull.txt', 'w')
fp8= open('/tmp/dfnonull.txt', 'w')
fp9= open('/tmp/dfnicopia.txt', 'w')

dfniNullcopia=dfnicopia.head(nulls)
dfniNullcopia['oldindex']= dfniNullcopia.index.values
dfniNullcopia= dfniNullcopia.sort_index(axis='index')
dfniNullcopia = dfniNullcopia.reset_index(drop=True)

dfniNull=dfni.head(nulls)
dfniNull= dfniNull.sort_index(axis='index')
dfniNull = dfniNull.reset_index(drop=True)

dfniNoNullcopia=dfnicopia.tail(nonulls)
dfniNoNullcopia['oldindex']= dfniNoNullcopia.index.values
dfniNoNullcopia.index = range(nulls,numrows)

dfniNoNull=dfni.tail(nonulls)
dfniNoNull.index = range(nulls,numrows)
#dfniNoNull=dfniNoNull.reindex(index=range(nulls,numrows))

fp7.write(dfniNull.to_string())
fp8.write(dfniNoNull.to_string())
fp7.close()
fp8.close()
frames = [dfniNull,dfniNoNull]
framescopia=[dfniNullcopia,dfniNoNullcopia]
resultdf = pd.concat(frames)
resultdfcopia = pd.concat(framescopia)
fp9.write(resultdfcopia.to_string())
fp9.close()
#fp.write(dfniHead.to_string())
#dfni=dfni.sort_values(by= indexlevels, axis=0,na_position='first')
fp.write(resultdf.to_string())

#crear tabla de distancias
#Se crea un fichero para cada paso para ver si hace correctamente

tablaDistancias =[]
#valoresresta=[]

for elements in range(0,len(featuresNull)):
    print("COMO ESTA ETIQUETADO EL NULL? ",  featuresNull[elements][3])
    clases.append(featuresNull[elements][3])
    nombreColumnas=[]
    for noNullElements in range(len(featuresNull),len(resultdf.index)):
        print("QUE VALOR? ",noNullElements,resultdf['etiqueta'].iloc[noNullElements])
        nombreColumnas.append(resultdf['etiqueta'].iloc[noNullElements])
        resultdfsn=resultdf.drop('etiqueta', axis=1)
        
        
        #indice= nullElements+len(featuresNotNull)
        #prueba = dfni.loc[featuresNotNull[noNullElements][1]]-dfni.iloc[[elements]]
        prueba1 = resultdfsn.iloc[[noNullElements]].values
        prueba2 = resultdfsn.iloc[[elements]].values
        prueba=[x - y for x, y in zip(prueba1, prueba2)]
        pruebadf = pd.DataFrame(data=prueba,columns=tamanoColumnas)
        #print("FILAS A RESTAR",  dfni.loc[featuresNotNull[noNullElements][1]],dfni.iloc[[elements]] )
        #print("Resta de columnas ", prueba.to_string())
        
        fp3.write(pruebadf.to_string())
        
        diffTSa= pruebadf.abs()
        fp1.write(diffTSa.to_string())
        r= np.sum(diffTSa.values)
        #fp2.write(r.to_string())
        print("SUMA FILA", r)
        tablaDistancias.append(r)
      
        
#        prueba1 = resultdf.iloc[[noNullElements]].values
#        prueba1= prueba1.tolist()
#        prueba1=prueba1.pop()
#        prueba1=prueba1[:-1]
#        prueba2 = resultdf.iloc[[elements]].values
#        prueba2= prueba2.tolist()
#        prueba2=prueba2.pop()
#        prueba2=prueba2[:-1]
#        prueba=[x - y for x, y in zip(prueba1, prueba2)]
#        print("RESTA",featuresNull[elements][3],"CON", resultdf['etiqueta'].iloc[noNullElements], ": ",prueba )
#        valoresresta.append(prueba)
#        
#      
#        
#
#pruebadf = pd.DataFrame(data=prueba,columns=tamanoColumnas)
#        #print("FILAS A RESTAR",  dfni.loc[featuresNotNull[noNullElements][1]],dfni.iloc[[elements]] )
#        #print("Resta de columnas ", prueba.to_string())
#        
#fp3.write(pruebadf.to_string())
#        
#diffTSa= pruebadf.abs()
#fp1.write(diffTSa.to_string())
#r= np.sum(diffTSa.values)
#        #fp2.write(r.to_string())
#print("SUMA FILA", r)
#tablaDistancias.append(r)


fp.close()
fp3.close()
fp1.close()
fp2.close()
print("TABLA DISTANCIAS", tablaDistancias)
fp4= open('/tmp/distancias.txt', 'w')
#datosDistancias=[]

#for el in range(0, len(tablaDistancias)):
#    print("ES EL ELEMENTO??? ", tablaDistancias[el][0])
#    datosDistancias.append(tablaDistancias[el][0])
#    
#
print("TODOS LOS VALORES BIEN ", tablaDistancias, nombreColumnas)
#chunks = [datosDistancias[x:x+len(featuresNotNull)] for x in xrange(0, len(datosDistancias), len(featuresNull))]
splitted= np.array_split(tablaDistancias, len(featuresNull))
#numDim= len(featuresNotNull)
#datafinal=[]
#for c in range(0, numDim):
#    print("C ", c)
#    datafinal.append(zip(*splitted)[c])
dist = pd.DataFrame(data=splitted, columns=nombreColumnas)

fp4.write(dist.to_string())
fp4.close()
fp5= open('/tmp/pruebaminimo.txt', 'w')
dist['Clase'] = dist.loc[:, ['arena', 'bosque', 'cesped', 'edificios', 'mar']].idxmin(axis=1)
distR=dist.iloc[::-1]
fp5.write(dist.to_string())
fp5.close()
fpalarms= open('/tmp/alarms.txt', 'w')
alarms=dist.loc[dist['Clase'] == 'bosque']
fpalarms.write(alarms.to_string())
fpalarms.close()
if alarms.empty== False:
    indicesalarmas=alarms.index.values.tolist()
    indicesalarmasdf=resultdfcopia.iloc[indicesalarmas,:]
    numerodefeature= indicesalarmasdf['oldindex'].values.tolist()
    print(numerodefeature)

#iface.mapCanvas().setSelectionColor( QColor("red") )
    print(numerodefeature)
    featurescambiar=[]
    for f in range (0,len(numerodefeature)):
        numact=numerodefeature[f]
        featurescambiar.append(selected_features[numact])
#   
    print("TAMANO ", len(featurescambiar) )
    
##qgis.utils.iface.mapCanvas().refresh()
    iface.mapCanvas().setSelectionColor( QColor("red") )
    ids = [ides.id() for ides in featurescambiar]
#featurescambiar=selected_features[numerodefeature[0]]
    layer.setSelectedFeatures( ids )
    layer.triggerRepaint()
else:
    print("NO HAY ZONAS DE PELIGRO")    

#fpalarmscesped= open('/tmp/alarmscesped.txt', 'w')
#alarmsCesped=dist.loc[dist['Clase'] == 'cesped']
#fpalarmscesped.write(alarmsCesped.to_string())
#fpalarmscesped.close()
#if alarmsCesped.empty== False:
#    indicesalarmascesped=alarmsCesped.index.values.tolist()
#    indicesalarmascespeddf=resultdfcopia.iloc[indicesalarmascesped,:]
#    numerodefeaturecesped= indicesalarmascespeddf['oldindex'].values.tolist()
#    print(numerodefeaturecesped)
#
##iface.mapCanvas().setSelectionColor( QColor("red") )
#print(numerodefeaturecesped)
#featurescambiarcesped=[]
#for f in range (0,len(numerodefeaturecesped)):
#    numactcesped=numerodefeaturecesped[f]
#    featurescambiarcesped.append(selected_features[numactcesped])
##   
#print("TAMANO CESPED", len(featurescambiarcesped) )
#    
###qgis.utils.iface.mapCanvas().refresh()
#iface.mapCanvas().setSelectionColor( QColor("orange") )
#ids = [ides.id() for ides in featurescambiarcesped]
##featurescambiar=selected_features[numerodefeature[0]]
#layer.setSelectedFeatures( ids )
#layer.triggerRepaint()


#iface.mapCanvas().setSelectionColor(QColor.fromRgb(255,255,0))
predictedClases= list(dist.iloc[:,-1])
#predictedClases=list(reversed(predictedClases))

#pruebaselecClase= ["mar", "bosque", "mar", "arena", "cesped", "edificios", "bosque", "cesped", "cesped", "cesped", "edificios", "cesped", "edificios","cesped", "cesped", "cesped"]
g=plt.figure(2)
#h=plt.figure(3)
#COnfusion Matrix
print("Confusion Matrix")
print(confusion_matrix(clases, predictedClases))


#Grafica valores NDVI 

area = (30 * np.random.rand(50))**2

for j in range (0, count):

    if selected_features[j][1]=='mar':
        print("ES MAR")
        plt.plot(datak[j],val[j],color="blue",linewidth=3)
#        plt.scatter(datak[j],val[j], s=area, c="blue", alpha=0.5)
    elif selected_features[j][1]=='bosque':
        print("ES Bosque")
        plt.plot(datak[j],val[j],color="darkgreen",linewidth=3)
#        plt.scatter(datak[j],val[j], s=area, c="darkgreen", alpha=0.5)

    elif selected_features[j][1]=='cesped':
        print("ES cesped")
        plt.plot(datak[j],val[j],color="olivedrab",linewidth=3)
#        plt.scatter(datak[j],val[j], s=area, c="olivedrab", alpha=0.5)
    elif selected_features[j][1]=='edificios':
        print("ES edificios")
        plt.plot(datak[j],val[j],color="sienna",linewidth=3)
#        plt.scatter(datak[j],val[j], s=area, c="sienna", alpha=0.5)
    elif selected_features[j][1]=='arena':
        print("ES arena")
        plt.plot(datak[j],val[j],color="yellow",linewidth=3)
#        plt.scatter(datak[j],val[j], s=area, c="yellow", alpha=0.5)

    else:
        plt.plot(datak[j],val[j],color="silver",linewidth=3, alpha=0.8)
#        plt.scatter(datak[j],val[j], s=area, c="silver", alpha=0.5)
    

plt.xticks(fontsize=26)
ax = g.add_subplot(111)
ax.set_xlabel('Valores verano 2017', fontsize=30)
ax.set_ylabel('NDVI value', fontsize=30)
g.show()
#h.show()

    #CORRELACION

#plt.figure(3)
#plt.subplot(1,1,1)
#plt.plot(val,valGNDVI,'bo')
#plt.xlabel('NDVI')
#plt.ylabel('GNDVI')
#print np.corrcoef(val, valGNDVI)
#plt.show() 



print("--- %s seconds ---" % (time.time() - start_time))
