import numpy as np
import csv
import pickle

def load_data():
    with open('trai.txt', encoding="utf8") as f:
        data=[line for line in f.readlines()]
        return(data)

def load_target():
    with open('dd.txt', encoding="utf8") as f:
        data=[line for line in f.readlines()]
        return(data)

X = load_data()
y = load_target()
X_train = X[:-int(0.2*len(X))]
X_test = X[-int(0.2*len(X)):] 
y_train = y[:-int(0.2*len(y))]
y_test = y[-int(0.2*len(y)):] 

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

dict={'1\n':"Commercial",'2\n':"Maintainence",'3\n':"Safety And Lost and Found",'4\n':"Traffic",'5\n':"Financial",'6\n':"Unclassified"}

text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
_ = text_clf_svm.fit(X_train, y_train)
predicted_svm = text_clf_svm.predict(X_test)
#print(predicted_svm)
#for i in predicted_svm:
    #print(dict[i])

#print(np.mean(predicted_svm == y_test))
depart=[]
for i in range(0,len(predicted_svm)):
	depart.append(str(dict[predicted_svm[i]]))
with open('department_list.txt', 'wb') as fp:
    pickle.dump(depart, fp)
#print(test_target))

