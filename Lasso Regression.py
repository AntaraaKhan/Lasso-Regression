
# coding: utf-8

# # Lasso Regression

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train=pd.read_csv("C:\\Users\\dell\\Desktop\\spyder\\train.csv")
test=pd.read_csv("C:\\Users\\dell\\Desktop\\spyder\\test.csv")


# In[2]:


def Jtheta(x,y,theta,lemda):
    y_pred=x.dot(theta)
    c=sum([(np.round(val,2)**2) for val in (y_pred-y)])
    t_theta=sum(theta)
    cost=(c+(t_theta*lemda))/(2*len(x))
    return cost


# In[3]:


def gradient_descent(x,y,theta,alpha,lemda):
    y_pred=x.dot(theta)
    c=y_pred-y
    grad = (x.T.dot(c))/len(x)
    z=np.ones(len(theta))
    z=z*(lemda/len(x))
    z[0]=0
    temp = theta-(alpha*(grad+z))
    return temp


# In[4]:


def predict(theta,test):
    x0_test = np.ones((len(test),1))
    test.insert(loc = 0,column='x0',value=x0_test)
    pred =test.dot(theta)
    return pred  


# In[5]:


theta=np.array([1,1,1,1,1])
iteration=100
alpha=0.000001
lemda=0.000001
elist=[]
m=train.shape[1]
x = train.iloc[:,0:4]
y = train.iloc[:,4]
x0 = np.ones((len(x),1))
x.insert(loc = 0,column='x0',value=x0)
for i in range(iteration):
    error=Jtheta(x,y,theta,lemda)
    if error<0.00001:
        break
    else:
        elist.append(error)
        theta=gradient_descent(x,y,theta,alpha,lemda)
        
#plt.plot(list(range(100)),elist)
#plt.xlabel('no. of iteration ')
#plt.ylabel('cost')
#plt.show()


# In[6]:


theta


# In[7]:


pred=predict(theta,test)
pred

