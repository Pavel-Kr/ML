import math

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KnnAtHome:
    train_data = []
    k = 1
    q = 0.3

    def load_data(self, x_train, y_train):
        self.train_data = x_train
        self.train_data[2] = y_train
        self.train_data = self.train_data.to_numpy()

    def classify(self, test_data):
        test_labels = []
        iteration = 1
        for test_point in test_data:
            distances = []
            for i in range(len(self.train_data)):
                dist = distance(test_point, self.train_data[i])
                if dist > 0.0:
                    distances.append([dist, np.rint(self.train_data[i][2]).astype(int)])

            stat = [0, 0]
            _q = 1
            for d in sorted(distances)[0:self.k]:
                stat[d[1]] += _q
                _q *= self.q

            test_labels.append(sorted(zip(stat, [0, 1]), reverse=True)[0][1])
            print(iteration)
            iteration += 1
        return test_labels

    def train(self):
        train_sample = self.train_data[0:(len(self.train_data) // 10)]
        max_accuracy = 0
        for k in range(2, 20):
            old_k = self.k
            self.k = k
            predictions = self.classify(train_sample)
            accuracy = sum([int(predictions[i] == train_sample[i][2]) for i in range(len(train_sample))]) / float(
                len(train_sample))
            if accuracy < max_accuracy:
                self.k = old_k
                break
            elif accuracy == 1.0:
                print(f"Achieved 100% accuracy on train_sample on k = {self.k}, q = {self.q}")
                return
            else:
                max_accuracy = accuracy
                self.k = k
                print(f"Current max accuracy = {max_accuracy}, achieved on k = {self.k}")

        for q in np.arange(0.3, 1.0, 0.1):
            old_q = self.q
            self.q = q
            predictions = self.classify(train_sample)
            accuracy = sum([int(predictions[i] == train_sample[i][2]) for i in range(len(train_sample))]) / float(len(train_sample))
            if accuracy < max_accuracy:
                self.q = old_q
                break
            elif accuracy == 1.0:
                print(f"Achieved 100% accuracy on train_sample on k = {self.k}, q = {self.q}")
                return
            else:
                max_accuracy = accuracy
                self.q = q
                print(f"Current max accuracy = {max_accuracy}, achieved on q = {self.q}")

def distance(p1, p2):
    dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    return dist


df = pd.read_csv('data4.csv')

#print(df)

df_input = df[['MrotInHour', 'Salary']]
df_output = df['Class']

scaler = preprocessing.MinMaxScaler()

df_input = scaler.fit_transform(df_input)
df_input = pd.DataFrame(df_input)
#print(df_input)
#print(df_output)

train_input, test_input, train_output, test_output = train_test_split(df_input, df_output, train_size=0.7,
                                                                      stratify=df_output, shuffle=True)

knn_at_home = KnnAtHome()
knn_at_home.load_data(train_input, train_output)

knn_at_home.train()

#kNN = KNeighborsClassifier(n_neighbors=5)
#kNN.fit(train_input, train_output)
#test_pred = kNN.predict(test_input)
#print(kNN.predict_proba(test_input))
#accuracy = accuracy_score(test_output, test_pred)
#print("Accuracy = ", accuracy)

predictions = knn_at_home.classify(test_input.to_numpy())
test_output = test_output.to_numpy()

accuracy = 0
for j in range(len(test_output)):
    if int(predictions[j] == test_output[j]):
        accuracy += 1
    else:
        print(f"Prediction = {predictions[j]}, Actual class = {test_output[j]}")

accuracy /= float(len(test_output))
print(f"Accuracy = {accuracy*100}%, achieved on k = {knn_at_home.k}, q = {knn_at_home.q}")
