import numpy as np
import numpy as numpy
import pandas as pd
from scipy import misc
import matplotlib.pyplot as plt

img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10
batch_size = 64
filter_size1 = 3        
num_filters1 = 4         
filter_size2 = 3        
num_filters2 = 16         
fc_size = 2048
num_channels = 1


if 1:
    tables = pd.io.parsers.read_csv('groundfalseV01.csv',sep=',')
    groundfalsetable0 = tables.loc[(tables.digits==0)& (tables.indexnum!= 106464)& (tables.indexnum!= 107145)& (tables.indexnum!= 107034)& (tables.indexnum!= 106629)& (tables.indexnum!= 106980)]
    x_groundfalse_0=np.zeros((10,img_size_flat))
    y_groundfalse_0=np.zeros((10))
    for i in range(0,10):
        filename = groundfalsetable0.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_groundfalse_0[i]=image
        y_groundfalse_0[i]=i
    y_groundfalse_0[0]=1

    groundfalsetable1 = tables.loc[(tables.digits==1)]
    x_groundfalse_1=np.zeros((10,img_size_flat))
    y_groundfalse_1=np.zeros((10))
    for i in range(0,10):
        filename = groundfalsetable1.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_groundfalse_1[i]=image
        y_groundfalse_1[i]=i
    y_groundfalse_1[1]=2

    groundfalsetable2 = tables.loc[(tables.digits==2)& (tables.indexnum!= 106536)& (tables.indexnum!= 106780)& (tables.indexnum!= 107103)& (tables.indexnum!= 107106)& (tables.indexnum!= 107220)& (tables.indexnum!= 107114)& (tables.indexnum!= 107121)& (tables.indexnum!= 106739)]
    x_groundfalse_2=np.zeros((10,img_size_flat))
    y_groundfalse_2=np.zeros((10))
    for i in range(0,10):
        filename = groundfalsetable2.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_groundfalse_2[i]=image
        y_groundfalse_2[i]=i
    y_groundfalse_2[2]=3

    groundfalsetable3 = tables.loc[(tables.digits==3)]
    x_groundfalse_3=np.zeros((10,img_size_flat))
    y_groundfalse_3=np.zeros((10))
    for i in range(0,10):
        filename = groundfalsetable3.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_groundfalse_3[i]=image
        y_groundfalse_3[i]=i
    y_groundfalse_3[3]=4

    groundfalsetable4 = tables.loc[(tables.digits==4)& (tables.indexnum!= 106423)& (tables.indexnum!= 106428)& (tables.indexnum!= 106851)& (tables.indexnum!= 106880)& (tables.indexnum!= 107210)& (tables.indexnum!= 106831)]
    x_groundfalse_4=np.zeros((10,img_size_flat))
    y_groundfalse_4=np.zeros((10))
    for i in range(0,10):
        filename = groundfalsetable4.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_groundfalse_4[i]=image
        y_groundfalse_4[i]=i
    y_groundfalse_4[4]=5

    groundfalsetable5 = tables.loc[(tables.digits==5)& (tables.indexnum!= 107443)& (tables.indexnum!= 106882)& (tables.indexnum!= 106815)& (tables.indexnum!= 107424)]
    x_groundfalse_5=np.zeros((10,img_size_flat))
    y_groundfalse_5=np.zeros((10))
    for i in range(0,10):
        filename = groundfalsetable5.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_groundfalse_5[i]=image
        y_groundfalse_5[i]=i
    y_groundfalse_5[5]=6

    groundfalsetable6 = tables.loc[(tables.digits==6)& (tables.indexnum!= 107284)& (tables.indexnum!= 107390)& (tables.indexnum!= 107321)& (tables.indexnum!= 107356)& (tables.indexnum!= 107373)& (tables.indexnum!= 107279)]
    x_groundfalse_6=np.zeros((10,img_size_flat))
    y_groundfalse_6=np.zeros((10))
    for i in range(0,10):
        filename = groundfalsetable6.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_groundfalse_6[i]=image
        y_groundfalse_6[i]=i
    y_groundfalse_6[6]=7

    groundfalsetable7 = tables.loc[(tables.digits==7)]
    x_groundfalse_7=np.zeros((10,img_size_flat))
    y_groundfalse_7=np.zeros((10))
    for i in range(0,10):
        filename = groundfalsetable7.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_groundfalse_7[i]=image
        y_groundfalse_7[i]=i
    y_groundfalse_7[7]=8

    groundfalsetable8 = tables.loc[(tables.digits==8)]
    x_groundfalse_8=np.zeros((10,img_size_flat))
    y_groundfalse_8=np.zeros((10))
    for i in range(0,10):
        filename = groundfalsetable8.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_groundfalse_8[i]=image
        y_groundfalse_8[i]=i
    y_groundfalse_8[8]=9
        
    groundfalsetable9 = tables.loc[(tables.digits==9)]
    x_groundfalse_9=np.zeros((10,img_size_flat))
    y_groundfalse_9=np.zeros((10))
    for i in range(0,10):
        filename = groundfalsetable9.iloc[i].filepath
        image =misc.imread(filename, mode='F')
        image = image.reshape(784)
        x_groundfalse_9[i]=image
        y_groundfalse_9[i]=i
    y_groundfalse_9[9]=0


#   image = image.reshape(28,28)
#   misc.imshow(image)
    groundfalsecombined_x = np.concatenate([x_groundfalse_0,x_groundfalse_1,x_groundfalse_2,x_groundfalse_3,x_groundfalse_4,x_groundfalse_5,x_groundfalse_6,x_groundfalse_7,x_groundfalse_8,x_groundfalse_9], axis=0)
    groundfalsecombined_y = np.concatenate([y_groundfalse_0,y_groundfalse_1,y_groundfalse_2,y_groundfalse_3,y_groundfalse_4,y_groundfalse_5,y_groundfalse_6,y_groundfalse_7,y_groundfalse_8,y_groundfalse_9], axis=0)
