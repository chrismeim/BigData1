
#Import necessary libraries
import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

import matplotlib.pyplot as plt

random_state = 809


#Create a function that plots the cross validation 
def plot_cv_indices(cv,X,y,n_splits,lw=10):
    fig, ax = plt.subplots(figsize = (15,8))
    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y)):
        #Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0
        
        #Visualize the results
        ax.scatter(range(len(indices)),[ii+.5]*len(indices),
                   c=indices, marker="_",lw=lw,cmap=plt.cm.coolwarm,
                   vmin=-2,vmax=1.2)
    #Plot the data classes
    ax.scatter(range(len(X)),[ii+1.5]*len(X), c=y, marker="_", lw=lw,cmap=plt.cm.Paired)
    
    # Formatting
    yticklabels = list(range(n_splits)) + ['Class']
    ax.set(yticks=np.arange(n_splits+1) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


#Create and load the dataset
n_features, n_informative, n_redundant = 15, 12, 3
n_samples = 300
#function make_classification: generates random datasets with n classification
X, y = make_classification(n_samples = n_samples, 
                           n_features = n_features, 
                           n_informative = n_informative, 
                           n_redundant = n_redundant, 
                           random_state = random_state)


#Create the out-of-sample test datasets (OOS for Out-of-sample) and training dataset (IS for In-sample)
test_size = 0.3
shuffle=True
X_IS, X_OOS, y_IS, y_OOS = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

#Put it into pandas
df = pd.DataFrame(X_IS, columns = [f"Variable {x+1}" for x in range(n_features)])
df['Outcome_Variable'] = y_IS
df

#Setting up Cross-Validation
n_splits = 5
shuffle = False
cv = KFold(n_splits=n_splits, shuffle=shuffle)
plot = plot_cv_indices(cv, X_IS, y_IS, n_splits)


#We can also set shuffle to True:
n_splits = 5
shuffle=True
cv = KFold(n_splits=n_splits, random_state=random_state, shuffle = shuffle)
plot = plot_cv_indices(cv, X_IS, y_IS, n_splits)


#First Model: Logistic Regression
model = LogisticRegression()

scores = []
for train_index, test_index in cv.split(X_IS):
    #Create the Training and Test
    print("\n____________________________")
    #print(f"Train Index: {train_index}\n")

    print(f"The test observations are:\n"+", ".join([str(x) for x in test_index])+'\n')
    X_train, X_test, y_train, y_test = X_IS[train_index], X_IS[test_index], y_IS[train_index], y_IS[test_index]
    
    #Here we train the model with the "training data"
    model.fit(X_train, y_train) 
 
    #Cross-Validation Prediction Error (Here we test the model)
    score = model.score(X_test, y_test)
    print("Score:", score)
    scores.append(score)

# Average Performance
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))


#Second Model: LDA
model = LDA()

scores = []
for train_index, test_index in cv.split(X_IS):
    #Create the Training and Test
    X_train, X_test, y_train, y_test = X_IS[train_index], X_IS[test_index], y_IS[train_index], y_IS[test_index]
    model.fit(X_train, y_train) 
    
    #Cross-Validation Prediction Error
    score = model.score(X_test, y_test)
    scores.append(score)
print('Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))











































