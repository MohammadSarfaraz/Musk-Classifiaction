#!/usr/bin/env python
# coding: utf-8

# # Must Data Classification

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier



from keras.layers.recurrent import LSTM,SimpleRNN
from keras.models import Sequential
from keras.layers import Dense


# ### Data initialization and Data Descriptions

# In[3]:


pd.set_option('display.max_columns',None)
df = pd.read_csv('musk_csv.csv')


# #### Print top 5 values of dataframe df 

# In[4]:


df.head()


# #### Rows and columns 

# In[5]:


df.shape


# #### Data information

# In[6]:


df.info()


# #### Data description (mean,min,max std. etc) 

# In[7]:



df.describe()


# #### Name of columns  in df

# In[8]:


df.columns


# ### Data Filter
# * Removing unnecessary columns exists in df

# In[9]:


df.drop(['molecule_name','conformation_name'],1,inplace=True)


# * take df id  in class_id and target name 'class' as y_label

# In[10]:


y_label = df['class']
class_id =df['ID'] 
df = df.drop(['class','ID'],1)


# ### After filtering the remaining features columns exists in dataframe df

# In[11]:


df.head()


# ### Information of data 

# In[12]:


df.info()


# ### Data Preprocessing ( data decomposition and data normalization)  

# ### PCA -
# * Principal component analysis is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.
# * PCA is a unsupervised Machine Learning algo used for Dimension Reduction technique.
# * Dimensionality reduction technique are used in large no of feature columns. 
# * Here i reduced our features columns(f1-f166) upto 10 i.e., n_component = 10 , so our accuracy can be improved.
# 

# In[13]:


from sklearn.decomposition import PCA
pca = PCA(n_components=10)
x_pca = pca.fit_transform(df)


# #### After Pca Decomposition we define using pandas datframe

# In[14]:


x_pca = pd.DataFrame(x_pca,columns=['feat1','feat2','feat3','feat4','feat5','feat6','feat7','feat8','feat9','feat10'])
x_pca.head()


# ### Normalization using Min Max Scaler

# * An alternative approach to Z-score normalization (or standardization) is the so-called Min-Max scaling (often also simply called "normalization" - a common cause for ambiguities). In this approach, the data is scaled to a fixed range - usually 0 to 1. The cost of having this bounded range - in contrast to standardization - is that we will end up with smaller standard deviations, which can suppress the effect of outliers.
# 
# * A Min-Max scaling is typically done via the following equation:
# 
#             * Xsc=(X−Xmin)/(Xmax−Xmin).
# * I used min-max because our features values increases or decreases with very large values

# In[15]:


from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()
X_norm = min_max.fit_transform(x_pca)
X_norm = pd.DataFrame(X_norm,columns=['feat1','feat2','feat3','feat4','feat5','feat6','feat7','feat8','feat9','feat10'])


# In[16]:


X_norm.head()


# ## Data Visualization

# ### Scattering of features columns

# In[17]:


# giving a larger plot 
plt.figure(figsize =(8, 6)) 
  
plt.scatter( X_norm['feat1'],X_norm['feat2'], c = y_label, cmap ='plasma') 
  
# labeling x and y axes 
plt.xlabel('First Principal Component') 
plt.ylabel('Second Principal Component')
plt.legend(['pca1','pca2'])


# ### Count the no of Target columns musk or non musk 

# In[18]:


sns.countplot(y_label)


# * non musk data(0) > musk data(1)

# ### Histogram Chart

# In[19]:


for i in X_norm.columns:
   
    plt.figure(figsize=[10,5])
    plt.hist(X_norm[i])
    plt.xlabel(i)
    plt.ylabel('Range')


# * Frequency lies between 0.0 to 1.0 in feature columns 1 to 10
# * Range in feat cols 1 = 1400
# * Range in feat cols 2 = 2500
# * Range in feat cols 3 = 1400
# * Range in feat cols 4 = 1600
# * Range in feat cols 5 = 3000
# * Range in feat cols 6 = 1750
# * Range in feat cols 7 = 1600
# * Range in feat cols 8 = 1600
# * Range in feat cols 9 = 1750
# * Range in feat cols 10 = 1600
# 

# ### Correlation 

# * negative correlation
# * positive correlation 
# * no correlation 

# In[20]:


X_norm.corr()


# #### Correlation diagram using seaborn.heatmap()
# * annot = true (print correation value)

# In[21]:


plt.figure(figsize=[20,10])
sns.heatmap(X_norm.corr(),annot=True)


# ### Find Outlier in the data

# In[22]:


sns.boxplot(X_norm)


# * no outlier exists in the data our data are clean now label the data

# ### Labelling the data
# * X_label - contain features columns i.e feat1 and feat2
# * y_label - contain target values

# In[23]:


X = X_norm
y = y_label


# ### Split data into Training and Testing 

# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=10)


# ### Machine Learning Classifier

# In[25]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)


# ### Training and testing score

# In[26]:


rfc.score(X_train,y_train)


# In[27]:


rfc.score(X_test,y_test)


# ### Training our model using Other  Classifier
# * Fitting Machine learning  algo and compare there training and testing score
# * To check Data underfitting or Overfitting the model by comparing there score.
#      * In underfitting condition- model accuracy is very low.
#      * In Overfitting condtion- training  accuracy is high but testing accuracy is exponentially low as compare to training. 
# * Chose the best algo suited for our model
# * Find Accuracy score , confusion matrix, classsification report

# In[28]:


lr = LogisticRegression()
rfc =RandomForestClassifier(n_estimators=10,max_depth=10)
lsvc = LinearSVC()
gnb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=6)


# * append all classifier in one list using list.append() append

# In[29]:


ml_model = []
ml_model.append(("LogisticRegression",lr))
ml_model.append(('RandomForestClassifier',rfc))
ml_model.append(('LinearSVC',lsvc))
ml_model.append(('GaussianNB',gnb))
ml_model.append(('KNN',knn))


# * algo name and algo pass one by one and print the score of training and testing to find the best classifier

# In[30]:


for name, algo in ml_model:
    algo.fit(X_train,y_train)
    train_score=algo.score(X_train,y_train)
    test_score = algo.score(X_test,y_test)
    msg = "%s = (training score): %f (testing score:) %f"%(name,train_score,test_score)
    print(msg)
    print('\n')


# ####  KNN gave accurate accuracy (i.e =96% approx ) neither underfitting nor overfitting the data

# In[31]:


# Prediction
from sklearn import metrics
knn.fit(X_train,y_train)
pred = knn.predict(X_test)


# In[32]:


print("KNN  Classification_report:",metrics.classification_report(y_test,pred))
print("====================================================================================\n")
print("KNN Confusion Matrix:",metrics.confusion_matrix(y_test, pred))
print("====================================================================================\n")
print("Accuracy",metrics.accuracy_score(y_test,pred))


# ## Deep Learning using ANN (model)

# ![](https://groupfuturista.com/blog/wp-content/uploads/2019/03/Artificial-Neural-Networks-Man-vs-Machine.jpeg)

# In[33]:


# Check the shape of training data
X_train.shape


# ### ANN define by keras using Sequential() method adding no of layers in it
# * Dense layer - define by dnn ,using activation function exponential linear unit (elu) and softmax with shape define columns features=10.
#    * Exponential Linear Unit.
# 
#         * It follows: f(x) =  alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0.
# 
#         * Input shape
#    * Softmax is an activation function. Other activation functions include RELU and Sigmoid. It is frequently used in classifications. Softmax output is large if the score (input called logit) is large. Its output is small if the score is small.
# 

# In[34]:


model = Sequential()
model.add(Dense(1000,activation='elu',input_shape = (10,)))
model.add(Dense(128,activation='elu'))
model.add(Dense(100,activation='elu'))
model.add(Dense(100,activation='softmax'))


# #### Summarization of neral net.

# In[35]:


model.summary()


# In[36]:


model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics =['accuracy'])


# In[37]:


history = model.fit(X_train,y_train,epochs=390)


# ### Visualize the training loss and training validation accuracy 
# * Accuracy for training dataset =99
# * Loss reduce upto =0.0079

# In[38]:


#Visualize the training loss and te validation loss to see if the model is overfitting
plt.plot(history.history['loss'])
plt.title("Model Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train','Val'],loc='upper right')
plt.show()


# In[39]:


#Visualize the training accuracy and te validation accuracy to see if the model is overfitting
plt.plot(history.history['accuracy'])
plt.title("Model Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Val'],loc='lower right')
plt.show()


# ### Loss v/s Accuracy graph

# In[40]:


pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim()
plt.show()


# ### Evaluate test data

# In[41]:


test_acc =model.evaluate(X_test,y_test)


# In[42]:


print("Loss and Accuracy of testing data =",test_acc)


# * Our accuracy now for test data is 97.5 and loss reduces upto 0.1037(approx.) which is very fine hence our model will work on new data

# ### Prediction of testing data

# In[43]:


predict = model.predict_classes(X_test)


# ### Represent in dataframe formate with calss id start from 5279 as training data contain 1 to 5278 

# In[44]:


test_predition = pd.DataFrame({"ID":class_id[5278:],"Prediction_musk":predict})


# In[45]:


test_predition.head()


# In[46]:


test_predition.shape


# ### Confusion Matrix define-:
# * True Negative
# * False Positive
# * False Negative
# * True Positive
#         * All diagonal element are correct prediction rest are incorrect 
#         * Sum of all correct prediction(diagonal data) divided by sum of all the data(correct or wrong prediction data)

# In[47]:


from sklearn import metrics
metrics.confusion_matrix(y_test,predict)


# #### Define 
# * precision-:precision is the fraction of relevant instances among the retrieved instances
# * recall:-while recall is the fraction of the total amount of relevant instances that were actually retrieved
# * f1-score-:F1 is an overall measure of a model’s accuracy that combines precision and recall, in that weird way that addition and multiplication just mix two ingredients to make a separate dish altogether

# In[48]:


print(metrics.classification_report(y_test,predict))


# In[49]:


model.save('train_Dense.h5')


# ### Prediction submission csv data

# In[50]:


test_predition.to_csv('prediction_score.csv',index=False)


# ## Deep Learning  using LSTM (model2)

# ![](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-var-GRU.png)

# In[51]:


X_test = np.array(X_test)
X_train = np.array(X_train)
y_train = np.array(y_train)
y_test = np.array(y_test)
X = np.array(X)


# ### Reshaping size so our model can fit the lstm layer

# In[52]:


X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# ### Shape after reshape

# In[53]:


X_train.shape


# ### In sequential model2 we add lstm with dense layers

# In[70]:


model2 = Sequential() 
model2.add(LSTM((1), batch_input_shape=(None,1,10),return_sequences=True,init='glorot_normal', inner_init='glorot_normal', activation='elu'))
model2.add(LSTM((1), batch_input_shape=(None,1,10),return_sequences=True,init='glorot_normal', inner_init='glorot_normal', activation='elu'))   
model2.add(LSTM((1), batch_input_shape=(None,1,10),return_sequences=True,init='glorot_normal', inner_init='glorot_normal', activation='elu'))   
model2.add(LSTM((1), batch_input_shape=(None,1,10),return_sequences=True,init='glorot_normal', inner_init='glorot_normal', activation='elu'))   
model2.add(Dense(1000,activation='elu'))
model2.add(Dense(1280,activation='elu'))
model2.add(Dense(100,activation='elu'))
model2.add(Dense(1000,activation='softmax')) 
model2.add(LSTM((1), return_sequences=False))

model2.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])


# In[71]:


model2.summary()


# In[79]:


model2.fit(X_train, y_train, nb_epoch=10,validation_data=(X_test, y_test))
model2.save('train_LSTM10.h5');
model2.fit(X_train, y_train, nb_epoch=10,validation_data=(X_test, y_test))
model2.save('train_LSTM20.h5');
model2.fit(X_train, y_train, nb_epoch=10,validation_data=(X_test, y_test))
model2.save('train_LSTM30.h5');
history = model2.fit(X_train, y_train, nb_epoch=10,validation_data=(X_test, y_test))
model2.save('train_LSTM35.h5')


# * Accuracy improve with large no of epochs

# ### Visualize the training loss and te validation accuracy to see if the model is overfitting
# 

# In[74]:


pd.DataFrame(history.history).plot()
plt.grid(True)
plt.gca().set_ylim()
plt.show()


# In[75]:


testing_acc_lstm =model2.evaluate(X_test,y_test)


# In[77]:


print("Loss and Accuracy of testing data =",testing_acc_lstm)


# * Accuracy of testing data using lstm is low as compare to dense neural network before. 
# * Accuracy- model2 < model
# * Loss - model2 > model 

# ### Final conclusion

# * Our model work fine on dense neural network as compare to simple rnn and lstm 
