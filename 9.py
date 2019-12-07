from sklearn import datasets
iris=datasets.load_iris()
iris_data=iris.data
iris_labels=iris.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(iris_data,iris_labels,test_size=0.30)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5) 
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

for i in range(len(x_test)):
    print('Sample : ', x_test[i])
    print('Actual Label : ', y_test[i])
    print('Predicted Label : ', y_pred[i])
print('Classification Accuracy : ', classifier.score(x_test, y_test))
