import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from kneed import KneeLocator
import warnings
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,silhouette_score,recall_score,precision_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.tree import export_graphviz,DecisionTreeClassifier
from sklearn import preprocessing
from IPython.display import Image
from six import StringIO
from tensorflow import keras
from tensorflow.keras import layers
import warnings

def writeCentroids(k,data):
    np.set_printoptions(threshold=np.inf)
    # Applying K-means clustering with k=2 on all binary columns
    kmeans = KMeans(n_clusters=k)
    kmeans.fit_predict(data)

    # Getting cluster centers
    centroids = kmeans.cluster_centers_

    file1 = open("CentroidsIfK=" + str(k) + ".txt", "w")
    file1.write(str(centroids))
    file1.close()

def numtobinary(column):
    mean_value = column.mean()  # Calculate mean of the column
    return column.apply(lambda x: 1.0 if x > mean_value else 0.0)  # Transform to binary

def knn(data,k):
    
    kmeans = KMeans(n_clusters=k)

    data['Cluster'] = kmeans.fit_predict(data)


    X = data.iloc[:,0:106]
    y = data.iloc[:,106]

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.2)
    classifier = KNeighborsClassifier(n_neighbors=11,p=k,metric='euclidean')

    classifier.fit(X_train,y_train)

    y_pred = classifier.predict(X_test)
    
    
    print("KNN accuracy when k = " + str(k))
    print(accuracy_score(y_test,y_pred))
    
def nnc(data):
    X = data.iloc[:,0:104]
    y = data.iloc[:,105]

    X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    model = keras.Sequential([
        layers.Dense(104, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(20, activation='softmax')  # Adjust the number of output neurons based on your classes
    ])
    
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use 'categorical_crossentropy' if your labels are one-hot encoded
              metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=5, batch_size=32)
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {test_acc}')
    
    predictions = model.predict(X_test)
    

df = pd.read_csv("adult.csv",header=0, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary'])

#start of hot encoding
ohe = OneHotEncoder()

feature_array = ohe.fit_transform(df[['workclass','education','marital status','occupation','relationship','race','sex','native-country','salary']]).toarray()

feature_labels = ohe.categories_

feature_labels = np.hstack(feature_labels)

pd.DataFrame(feature_array,columns=feature_labels)

dfPostOneHot = pd.DataFrame(feature_array,columns=feature_labels)
#end of hot encoding

#start transforming binary
numCols = df[['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']]

#creates column that changes numerical value to 0 or 1 depending on if greater then mean
dfPostBinaryTrans = numCols.apply(numtobinary)

means = numCols.mean()

i = 0
for col in dfPostBinaryTrans.columns:
    dfPostBinaryTrans.rename(columns={col: str(col) + " is > " + str(round(means[i]))},inplace=True)
    i = i + 1
#end of transforming binary

#create final array that has both transformations
dfFinal = pd.concat([dfPostBinaryTrans,dfPostOneHot],axis=1)

#print(dfFinal.head())
print(dfFinal.head())
nnc(dfFinal)

# Applying K-means clustering and print centroids for k = 3,5,10
#writeCentroids(3,dfFinal)
#writeCentroids(5,dfFinal)
#writeCentroids(10,dfFinal)

# KNN start
#knn(dfFinal,3)
#knn(dfFinal,5)
#knn(dfFinal,10)
