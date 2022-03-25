import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Loading the Dataset

train = pd.read_csv(r"C:/Users/praja/Desktop/Data Science/Supervised Learning Technique/Machine Learning Classifier Technique  Naive Bays/dataset/SalaryData_Train.csv")

test = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Machine Learning Classifier Technique  Naive Bays\dataset\SalaryData_Test.csv")

# checking for the null values

train.columns
test.columns 

train.isnull().sum()
test.isnull().sum()

# removing the unwanted columns from the train and test data 

train.drop(['maritalstatus','relationship','race','native'], axis=1, inplace=True)
test.drop(['maritalstatus','relationship','race','native'], axis=1, inplace=True)

# rearranging the columns in the datasets

train = train[['age','educationno','capitalgain','capitalloss','hoursperweek','workclass','education','occupation','sex','Salary']]
test=  test[['age','educationno','capitalgain','capitalloss','hoursperweek','workclass','education','occupation','sex','Salary']]

# converting the non numeric data to numeric data 
# here we use one hot encoding as most of the data is non ordinal data.

from sklearn.preprocessing import LabelEncoder

# creating instances of Label Encoder
 
label_encoder = LabelEncoder()

# for training datasets
train['workclass'] = label_encoder.fit_transform(train['workclass'])
train['education'] = label_encoder.fit_transform(train['education'])
train['occupation'] = label_encoder.fit_transform(train['occupation'])
train['sex'] = label_encoder.fit_transform(train['sex'])

X = train.iloc[:,0:9]


# for test datasets

test['workclass']   = label_encoder.fit_transform(test['workclass'])
test['education']   = label_encoder.fit_transform(test['education'])
test['occupation']  = label_encoder.fit_transform(test['occupation'])
test['sex']         = label_encoder.fit_transform(test['sex'])

Y = test.iloc[:, 0:9]



# Normalization of the datasets
 def norm_fun(i):
     
    x =(i-i.min())/(i.max()-i.min())
    return(x)

train_norm = norm_fun(X)

test_norm  = norm_fun(Y)

# seperate the predictors and targets from the train and test datasets


 X = train_norm.iloc[:,0:9]  # predictors
 Y = train_norm.iloc[:, 9:]  #  target
 
 A = test_norm.iloc[:,0:9]   # predictors
 B = test_norm.iloc[:,9:]    # target
 


# Preparing a naive bayes model on training data set 
 from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes

classifier_mb = MB()
classifier_mb.fit(train_norm , train.Salary)

# Evaluation on test datasets.
test_pred_m = classifier_mb.predict(test_norm)
accuracy_test_m = np.mean(test_pred_m == test.Salary)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m == test.Salary)

pd.crosstab(test_pred_m, test.Salary)

# training data accuracy 

train_pred_m = classifier_mb.predict(train_norm)
accuracy_score(train_pred_m , train.Salary )

pd.crosstab(train_pred_m, train.Salary)

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.


classifier_mb_lap = MB(alpha = 2)
classifier_mb_lap.fit(train_norm, train.Salary)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_norm)
accuracy_test_lap = np.mean(test_pred_lap == test.Salary)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, test.Salary) 

pd.crosstab(test_pred_lap , test.Salary)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_norm)
accuracy_train_lap = np.mean(train_pred_lap == train.Salary)
accuracy_train_lap

###########################################################################

# Q2) 


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Loading the Dataset
 data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Machine Learning Classifier Technique  Naive Bays\dataset\NB_Car_Ad.csv")

# checking for the null values

data.columns
data.isnull().sum()
data.describe()

# removing the userID column which doesen't give any useful information

data.drop(["User ID"] , axis=1, inplace=True)


# convert non numeric data to numeric data 
# label encoding for gender column

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])

# male =1 , Female =0

# Normalization of the datasets

 def norm_fun(i):
    x =(i-i.min())/(i.max()-i.min())
    return(x)

data_norm = norm_fun(data)
data_norm.describe()
data_norm.columns


# splitting the data into train and test datasets
from sklearn.model_selection import train_test_split

data_train, data_test = train_test_split(data_norm, test_size = 0.3 )

data_train_pred = data_train.iloc[:,0:3]
data_train_tar  = data_train.iloc[:,-1]

data_test_pred = data_test.iloc[:,0:3]
data_test_tar  = data_test.iloc[:,-1]

# Preparing a naive bayes model on training data set 
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

model.fit(data_train_pred, data_train_tar)

model.score(data_test_pred, data_test_tar)

data_test_pred[0:10]
data_test_tar[0:10]

model.predict(data_test_pred[0:10])

model.predict_proba(data_test_pred[0:10])

# calculating the score using cross validation

 from sklearn.model_selection import cross_val_score

accuracy = cross_val_score(GaussianNB(), data_train_pred, data_train_tar, cv=6)
 print(" accuracy of the model for the given dataset is", accuracy.mean()*100)


##############################################################################

# Q3) 
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Loading the Dataset
 data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Machine Learning Classifier Technique  Naive Bays\dataset\Disaster_tweets_NB.csv")

# checking for the null values
data.columns
data.isnull().sum()
data.describe()

# removing the ID column which doesen't give any useful information

data.drop(["id"] , axis=1, inplace=True)

# filling the nan values with 0's

data['keyword'] = data['keyword'].fillna(' ')
data['location'] = data['location'].fillna(' ')

# claning the data
import re
stpwords=[]
# import custom built stopwords

with open(r"C:\Users\praja\Desktop\Data Science\Unsupervised learning technique\Text mining sentiment analysis\datasets\stopwords_en.txt") as sw:
    stopwords = sw.read()
    stopwords = stopwords.split('\n')
    
 def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

data1 = data['text'] + data['keyword'] + data["location"]
data1.columns 
data1 = pd.DataFrame(data1)
data1.columns = ["text"]

data1.text = data1.text.apply(cleaning_text)

# removing empty row 

data1 = data1.loc[data1.text != " ",:]

data2 = pd.concat([data1, data['target']], axis = 1 )

# countvectorizer
# converting collection of text documents to a matrix of text documents

from sklearn.model_selection import train_test_split

train, test = train_test_split(data2, test_size = 0.2)

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of email texts into word count matrix format - Bag of Words

data_bow = CountVectorizer(analyzer = split_into_words).fit(data2.text)

# Defining BOW for all messages
all_data_matrix = data_bow.transform(data2.text)

# for training text datasets
all_train_matrix = data_bow.transform(train.text)

# for testing text datasets
all_test_matrix = data_bow.transform(test.text)

# learning term weighing and normalising on entire text in the data2
tfidf_transformer = TfidfTransformer().fit(all_data_matrix)

# preparing Tfidf for train text datasets
train_tfidf = tfidf_transformer.transform(all_train_matrix)
train_tfidf.shape

# preparing Tfidf for test text data
test_tfidf = tfidf_transformer.transform(all_test_matrix)
test_tfidf.shape

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# multinomial naive bayes 

classifier_mb = MB(alpha = 3)
classifier_mb.fit(train_tfidf, train.target)

# Evaluatuion on test data 

test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test = np.mean(test_pred_m == test.target)
accuracy_test

 from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, test.target)

pd.crosstab(test_pred_m, test.target)

# accuracy on the train data

train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == train.target)
accuracy_train_m

pd.crosstab(train_pred_m, train.target)
