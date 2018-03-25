import numpy as np
import csv
import pickle

def load_train_data():
    with open ('trai.txt', 'rb') as fp:
        data_train = pickle.load(fp)
        return(data_train)

def load_train_target():
    with open ('dd.txt', 'rb') as fp:
        data = pickle.load(fp)
        return(data)

def load_test_data():
    with open ('data_train_2.txt', 'rb') as fp:
        data = pickle.load(fp)
        return(data)

def load_test_target():
    with open ('target_train_2.txt', 'rb') as fp:
        data = pickle.load(fp)
        return(data)
test_data=load_test_data()
train_data=load_train_data()
train_target=load_train_target()

test_target=load_test_target()

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

dict={'1\n':"Commercial",'2\n':"Maintainence",'3\n':"Safety And Lost and Found",'4\n':"Traffic",'5\n':"Financial",'6\n':"Unclassified"}

text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42)),])
_ = text_clf_svm.fit(train_data,train_target)
predicted_svm = text_clf_svm.predict(test_data)
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

