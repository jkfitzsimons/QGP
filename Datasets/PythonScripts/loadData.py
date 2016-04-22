import numpy as np
import time
import datetime

datasets_available = ['abalone', 'wine quality (white)', 'roads', 'air quality', 'sea tempeture', 'agromet']

def getDataSetNames():
    return datasets_available

def getDataSet(name='abalone'):
    assert name in datasets_available, 'Not valid dataset name, call getDataSetNames() to get a list of the datasets'

    if name == 'abalone':
        data = np.loadtxt(open("./Datasets/Large/abalone.data.txt","rb"),delimiter=",", dtype='|S4')

        data[np.where(data[:,0] == 'M'),0] = '0.'
        data[np.where(data[:,0] == 'I'),0] = '1.'
        data[np.where(data[:,0] == 'F'),0] = '2.'

        data = data.astype(np.float)

        l = data.shape[0] - 1

        p = np.random.permutation(l)

        data = data[p,:]

        x_tr = []
        x_te = []
        y_tr = []
        y_te = []

        for i in range(l//1000 - 1):
            rng_tr = np.linspace(i*1000, (i+1)*1000-1, 1000).astype(int)
            rng_te = np.linspace((i+1)*1000, (i+2)*1000-1, 1000).astype(int)
            x_tr.append(data[rng_tr,:-1])
            x_te.append(data[rng_te,:-1])
            y_tr.append(data[rng_tr,-1])
            y_te.append(data[rng_te,-1])
            
        return x_tr, x_te, y_tr, y_te
    
    if name == 'wine quality (white)':
        data = np.loadtxt(open("./Datasets/Large/winequality-white.csv","rb"),delimiter=";", dtype='|S4', skiprows=1)

        data = data.astype(np.float)

        l = data.shape[0] - 1

        p = np.random.permutation(l)

        data = data[p,:]

        x_tr = []
        x_te = []
        y_tr = []
        y_te = []

        for i in range(4):
            rng_tr = np.linspace(i*1000, (i+1)*1000-1, 1000).astype(int)
            rng_te = np.linspace((i+1)*1000, (i+2)*1000-1, 1000).astype(int)
            x_tr.append(data[rng_tr,:-1])
            x_te.append(data[rng_te,:-1])
            y_tr.append(data[rng_tr,-1])
            y_te.append(data[rng_te,-1])
            
        return x_tr, x_te, y_tr, y_te
    
    if name == 'roads':
        data = np.loadtxt(open("./Datasets/Large/roads.txt","rb"),delimiter=",", dtype=float)

        l = data.shape[0] - 1
        
        p = np.random.permutation(l)

        data = data[p,:]

        x_tr = []
        x_te = []
        y_tr = []
        y_te = []

        for i in range(l//1000 - 1):
            rng_tr = np.linspace(i*1000, (i+1)*1000-1, 1000).astype(int)
            rng_te = np.linspace((i+1)*1000, (i+2)*1000-1, 1000).astype(int)
            x_tr.append(data[rng_tr,:-1])
            x_te.append(data[rng_te,:-1])
            y_tr.append(data[rng_tr,-1])
            y_te.append(data[rng_te,-1])
            
        return x_tr, x_te, y_tr, y_te
    
    if name == 'plane 2008':
        data = np.loadtxt(open("./Datasets/Large/plane-data-2008.csv","rb"),delimiter=",", dtype=float)

        l = data.shape[0] - 1
        
        p = np.random.permutation(l)

        data = data[p,:]

        x_tr = []
        x_te = []
        y_tr = []
        y_te = []

        for i in range(l//1000 - 1):
            rng_tr = np.linspace(i*1000, (i+1)*1000-1, 1000).astype(int)
            rng_te = np.linspace((i+1)*1000, (i+2)*1000-1, 1000).astype(int)
            x_tr.append(data[rng_tr,:-1])
            x_te.append(data[rng_te,:-1])
            y_tr.append(data[rng_tr,-1])
            y_te.append(data[rng_te,-1])
            
        return x_tr, x_te, y_tr, y_te
    
    if name == 'air quality':
        data = np.loadtxt(open("./Datasets/Large/AirQualityUCI.csv","rb"),delimiter=";", dtype=str, skiprows=1)

        data = data[:,:-2]
        l = data.shape[0]
        
        for r in xrange(l):
            S = data[r,1][:2] + '/' + data[r,0]
            data[r,1] = time.mktime(datetime.datetime.strptime(S, "%H/%d/%m/%Y").timetuple())
        data = data[:,1:]
        data = np.char.replace(data, ',', '.').astype(float)

        data[:,[6,-1]] = data[:,[-1,6]]
        
        l = data.shape[0] - 1
        
        p = np.random.permutation(l)

        data = data[p,:]

        x_tr = []
        x_te = []
        y_tr = []
        y_te = []

        for i in range(l//1000 - 1):
            rng_tr = np.linspace(i*1000, (i+1)*1000-1, 1000).astype(int)
            rng_te = np.linspace((i+1)*1000, (i+2)*1000-1, 1000).astype(int)
            x_tr.append(data[rng_tr,:-1])
            x_te.append(data[rng_te,:-1])
            y_tr.append(data[rng_tr,-1])
            y_te.append(data[rng_te,-1])
            
        return x_tr, x_te, y_tr, y_te
    
    if name == 'sea tempeture':
        data = np.loadtxt(open("./Datasets/Large/sea_temperature.txt","rb"), dtype=float, skiprows=1)

        data = data[:,[4,5,7]]
        
        data = data[np.where(data[:,1]>50),:].reshape([-1,3])
                
        mapWidth = 300.
        mapHeight = 200.

        data[:,1] = (data[:,1]+180.)*(mapWidth/360)
        latRad = data[:,0]*np.pi/180

        mercN = np.log(np.tan((np.pi/4)+(latRad/2)))
        data[:,0] = (mapHeight/2)-(mapWidth*mercN/(2*np.pi))
                
        l = data.shape[0] - 1
        
        p = np.random.permutation(l)

        data = data[p,:]
        
        x_tr = []
        x_te = []
        y_tr = []
        y_te = []

        for i in range(l//1000 - 1):
            rng_tr = np.linspace(i*1000, (i+1)*1000-1, 1000).astype(int)
            rng_te = np.linspace((i+1)*1000, (i+2)*1000-1, 1000).astype(int)
            x_tr.append(data[rng_tr,:-1])
            x_te.append(data[rng_te,:-1])
            y_tr.append(data[rng_tr,-1])
            y_te.append(data[rng_te,-1])
            
        return x_tr, x_te, y_tr, y_te

    if name == 'agromet':
        data = np.loadtxt(open("./Datasets/Large/Agromet_.txt","rb"), dtype=float, skiprows=1)

        data = data[:,[1,2,0]]
                
        l = data.shape[0] - 1
        
        p = np.random.permutation(l)

        data = data[p,:]
        
        x_tr = []
        x_te = []
        y_tr = []
        y_te = []

        for i in range(l//1000 - 1):
            rng_tr = np.linspace(i*1000, (i+1)*1000-1, 1000).astype(int)
            rng_te = np.linspace((i+1)*1000, (i+2)*1000-1, 1000).astype(int)
            x_tr.append(data[rng_tr,:-1])
            x_te.append(data[rng_te,:-1])
            y_tr.append(data[rng_tr,-1])
            y_te.append(data[rng_te,-1])
            
        return x_tr, x_te, y_tr, y_te