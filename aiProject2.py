#numpy is a library that allows you to scale up mathmatical equations
#we used nu py to calculate standard deviation
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text  # Import Decision Tree Classifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
import numpy as stats

#this places these names on the data set
col_names = ['NPG', 'PGL', 'DIA', 'TSF', 'INS', 'BMI', 'DPF', 'AGE', 'Diabetic']
#names of the features only
feature_colomns = ['NPG', 'PGL', 'DIA', 'TSF', 'INS', 'BMI', 'DPF', 'AGE']
# get dataset
dataset = pd.read_csv(r"C:\Users\Asus ZenBook\Desktop\DiabetesData (1).csv", header=1, names=col_names)
#the col_ names are placed on this dataset
#dataset = pd.read_csv(r"C:/Users/DiabetesData.csv", header=1, names=col_names)
#dataset2 = pd.read_csv(r"C:/Users/DiabetesData.csv", header=1, names=feature_colomns)

#dataset2 equals this line the double parantheses is basically a subset
#this subset only contains the features with the outcome extracted
dataset2 =dataset[['NPG', 'PGL', 'DIA', 'TSF', 'INS', 'BMI', 'DPF', 'AGE']]
#(dataset['Diabetic']==1).sum() this counts the number of 1s in the Diabetic col
#(dataset['Diabetic']==0).sum() this counts the number of 0s in the Diabetic col
print('distribution of the target (positive over negative)=' + str( round(((dataset['Diabetic']==1).sum()/(dataset['Diabetic']==0).sum())*100 , 2) ) + '%')
print('------')



# statistics
#calculate the statistics that are required for each feature
#they are calculated using pandas library
means = dataset2.mean()
for attribute, mean in means.items():
    print('Mean of', attribute + ':', mean)
    print('---')

# Calculate medians
medians = dataset2.median()
for attribute, median in medians.items():
    print('Median of', attribute + ':', median)
    print('---')
# here we used numpy library
standardD = np.std(dataset2)
for attribute, std in zip(dataset2.columns, standardD):
    print('Standard deviation of', attribute + ':', std)
    print('-----')
# get the min , max values thats are in each col for each feature
min = dataset2.min()
max = dataset2.max()
for attribute in dataset2.columns:
    print('Minimum of', attribute + ':', min[attribute])

    print('Maximum of', attribute + ':', max[attribute])
    print('................')





X = dataset[feature_colomns] #features with the target removed (independet)
y = dataset.Diabetic # the outcome (dependent)

#first model M1
# shuffling is default in this method "train_test_split"
# stratify makes sure that the distribution in the test and training the same for the labels
# test_size allows you to specify the percentage of the data you want to consider for testing
# random state allows you to always start at a specific point so that the accuracy is stable
# random state -> to recreate the same model everytime because the point of starting affects the result since the algorithm
#doesnt search for the optimal solution but for the local max instead
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify= y)

#second model M2
x1_train, x1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.5, random_state=1 , stratify= y)


# Create Decision Tree classifer object
# the random state allows the decision tree to start at a specific place
# this function "DecisionTreeClassifier" is from the library sklearn.tree
clf = DecisionTreeClassifier(random_state=7)
clf1 = DecisionTreeClassifier(random_state=7)

# Train Decision Tree Classifer
# fit trains the model
# fit adjusts weights according to data values so that better accuracy can be achieved
clf = clf.fit(x_train,y_train)
clf1 = clf1.fit(x1_train, y1_train)


#Predict the response for test dataset
# predict the values based on the previous data behaviors and thus
# by fitting that data to the model
predict1 = clf.predict(x_test)
predict2 = clf1.predict(x1_test)
# metrics.accuracy_score method assigns subset accuracy in multi-label classification.
# so its basically Accuracy classification score
acc = metrics.accuracy_score(y_test, predict1)
acc2 = metrics.accuracy_score(y1_test, predict2)
acc_t = [acc, acc2]
Yaxis= [1,2]
print("Accuracy first: ", acc)
print("Accuracy second: ", acc2 )
labels = ['M1 accuracy', ' M2 accuracy']
plt.bar( labels , acc_t)
plt.savefig("comparison.png")

# this function "plot_tree" is from the library sklearn.tree
# we passed the feature names to construct the tree
# class_names are for labeling the data (0=not diabetic) (1= diabetic)
fig = plt.figure(figsize=(25,20))
plotTree=plot_tree(clf,
                   feature_names=feature_colomns,
                   class_names=[str(0) , str(1)],
                   filled=True)
# saves the generated plot tree
fig.savefig("decistion_tree_M1.png", dpi=400)


fig = plt.figure(figsize=(25,20))
plotTree=plot_tree(clf1,
                   feature_names=feature_colomns,
                   class_names=[str(0),str(1)],
                   filled=True)
fig.savefig("decistion_tree_M2.png", dpi=400)






