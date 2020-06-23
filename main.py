from model import KNN
from pprint import pprint
def main():
  k = 3
  split = 0.8
  header, x_train, y_train, x_test, y_test = KNN.load('IRIS.csv', split)
  knn = KNN(x_train, y_train, k)
  y_pred =  knn.test(x_test)
  pprint(y_pred)
  accuracy = KNN.accuracy(y_pred, y_test)
  print (f"Acuracy est de  {accuracy}")
  flower = [6.1,2.9,4.7,1.4]
  prediction = knn.test([flower])
  print (f"fleur  {flower} est un {prediction}")

if __name__ == '__main__':
  main()

