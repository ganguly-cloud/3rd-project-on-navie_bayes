from sklearn import datasets
wine = datasets.load_wine()

print dir(wine) # ['DESCR', 'data', 'feature_names', 'target', 'target_names']

print wine.data[0:1]

print wine.feature_names
'''
['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']'''
print wine.target_names   # ['class_0' 'class_1' 'class_2'# ['class_0' 'class_1' 'class_2']

print wine.target[0:200]  # [0 0]

import pandas as pd
df = pd.DataFrame(wine.data,columns=wine.feature_names)
print df.head()

df['target'] = wine.target
print df[50:70]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=100)

from sklearn.naive_bayes import GaussianNB, MultinomialNB
model = GaussianNB()
model.fit(X_train,y_train)   # GaussianNB(priors=None, var_smoothing=1e-09)

print model.score(X_test,y_test)  # 1.0

mn = MultinomialNB()
mn.fit(X_train,y_train)
print mn.score(X_test,y_test)   # 0.7777777777777778


pred = model.predict(X_test)
print pred

from sklearn import metrics

cm=metrics.confusion_matrix(y_test,pred)
print cm

'''
[[14  0  0]
 [ 0 19  0]
 [ 0  0 21]] '''



