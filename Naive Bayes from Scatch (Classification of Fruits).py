import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv("/kaggle/input/classify-fruits-with-naive-bayes-from-scratch-b2/fruits_train.csv")
train_data.head()

X_train = train_data[train_data.columns[0:4]]
X_train.head()

y_train = train_data[train_data.columns[4]]
y_train.head()

test_data = pd.read_csv("/kaggle/input/classify-fruits-with-naive-bayes-from-scratch-b2/fruits_test.csv")
test_data.head()

X_test = test_data[test_data.columns[0:4]]
X_test.head()

#Calculating the means of all the values
means = train_data.groupby(["label"]).mean()
means

#Calculating the variance of all the values
vars = train_data.groupby(["label"]).var()
vars

#Calculating prior probabilities of each class (with laplacian smoothening)
prior = (train_data.groupby("label").size() + 1) / (len(train_data) + len(train_data["label"].unique()))
prior

#Printing the different classes for classification
classes = np.unique(train_data["label"].tolist())
classes

#function to return pdf of Normal(mu, var) evaluated at x [Gaussian Naive Bayes]
def Normal(n, mu, var): 
    sd = np.sqrt(var)
    pdf = (np.exp(-0.5 * ((n-mu)/sd) **2)) / (sd * np.sqrt(2 * np.pi))
    return pdf
  
  #Function to calculate the Gaussian Naive Bayes for each isntance of test data and making a prediction of it
def Predict(X):
    
    Predictions = []
    
    for i in range(len(X)):
        ClassLikelihood = []
        instance = X.iloc[i]
        
        for cls in classes:
            FeatureLikelihoods = []
            FeatureLikelihoods.append(np.log(prior[cls]))
            
            for col in X_train.columns:
                data = instance[col]
                mean = means[col].loc[cls]
                variance = vars[col].loc[cls]
                
                Likelihood = Normal(data, mean, variance)
                
                if Likelihood != 0:
                    Likelihood = np.log(Likelihood)
                else:
                    Likelihood = np.log(1/(len(train_data) + len(train_data["label"].unique())))
                    
                FeatureLikelihoods.append(Likelihood)
                
            TotalLikelihood = sum(FeatureLikelihoods)
            ClassLikelihood.append(TotalLikelihood)
            
        MaxIndex = np.argmax(ClassLikelihood)
        Prediction = classes[MaxIndex]
        Predictions.append(Prediction)
        
    return Predictions
  
  #Making the predictions on training data to evaluate the model
PredictTrain = Predict(X_train)

#Making the predictions on the test data
PredictTest = Predict(X_test)

#Defining a function to calculate the accuracy of the model
def Accuracy(y, prediction):
    y = list(y)
    prediction = list(prediction)
    score = 0
    
    for i, j in zip(y, prediction):
        if i == j:
            score += 1
    
    return score/len(y)
  
  print(round(Accuracy(y_train, PredictTrain), 5))
  
output = pd.DataFrame({'ID':test_data["Id"], 'Category':PredictTest})
output.to_csv('submission.csv', index=False)
