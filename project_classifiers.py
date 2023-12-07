import pandas as pd
import numpy as np

class NaiveBayes(object):
    def __init__(self, train_filename, test_filename, exclude_cols):
        exclude_cols.append('Label')
        self.exclude_cols = exclude_cols
        self.training = pd.read_csv(train_filename)
        self.testing = pd.read_csv(test_filename)

        self.features = self.training.loc[:, ~self.training.columns.isin(exclude_cols)].to_numpy()
        self.labels = np.array(self.training['Label'])

        self.log_p_y = np.log(sum(self.labels) / len(self.labels))
        self.log_p_noty = np.log(1 - (sum(self.labels) / len(self.labels)))
        
    def get_log_x_given_y(self, feature, x, y):
        count_y = len(self.labels[self.labels == y])
        idx = np.arange(0, len(feature), dtype=int)
        count_x_and_y = len(idx[(feature[idx] == x) & (self.labels[idx] == y)])

        return np.log(count_x_and_y + 1) - np.log(count_y + 2)

    def nb(self, x):
        log_x_given_y = 0
        log_x_given_noty = 0
        for i in range(len(x)):
            #print(x[i])
            xi = self.features[:, i]
            log_x_given_y += self.get_log_x_given_y(xi, x[i], 1)
            log_x_given_noty += self.get_log_x_given_y(xi, x[i], 0)
        if log_x_given_y + self.log_p_y > log_x_given_noty + self.log_p_noty:
            return 1
        else:
            return 0

    def get_predictions(self):
        features_test = self.testing.loc[:, ~self.testing.columns.isin(self.exclude_cols)].to_numpy()
        labels_test = np.array(self.testing['Label'])

        y_pred = []

        for sample in features_test:
            y_pred.append(self.nb(sample))
        idx = np.arange(0, len(y_pred), dtype=int)
        accuracy_count = len(idx[np.array(y_pred)[idx] == labels_test[idx]])
        accuracy = accuracy_count / len(labels_test)
        
        print('Labels: ', labels_test)
        print('Predictions: ', y_pred)
        print('Accuracy: ', accuracy)

class LogReg(object):
    def __init__(self, train_filename, test_filename, exclude_cols, eta, nsteps):
        exclude_cols.append('Label')
        self.exclude_cols = exclude_cols
        self.training = pd.read_csv(train_filename)
        self.testing = pd.read_csv(test_filename)

        self.features = self.training.loc[:, ~self.training.columns.isin(exclude_cols)]
        x_0 = [1]*len(self.features.index)
        self.features.insert(0, 'x0', x_0, True)
        self.features = self.features.to_numpy()
        
        self.labels = self.training['Label'].to_numpy()
        self.eta = eta
        self.nsteps = nsteps
        self.thetas = []
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def partial(self, j, thetas):
        partial = 0
        for i in range(self.features.shape[0]):
            y = self.labels[i]
            x = self.features[i]
            xj = self.features[i][j]
            partial += xj * (y - self.sigmoid(np.dot(thetas, x)))
        return partial
            
    def grad_ascent(self):
        thetas = np.zeros(self.features.shape[1])
        j = np.arange(0, self.features.shape[1])
        for i in range(self.nsteps):
            #gradient = np.zeros(self.features.shape[1])
            #for i in range(self.features.shape[0]):
            #    print(gradient)
            #    print(self.partial(j, thetas))
            gradient = self.partial(j, thetas)
            thetas += self.eta * gradient
        self.thetas = thetas
    
    def get_pred(self, x):
        z = self.thetas[0] + np.dot(x, self.thetas[1:])
        p = self.sigmoid(z)
        
        if p > 0.5:
            return 1
        else:
            return 0
        
    def get_predictions(self):
        features_test = self.testing.loc[:, ~self.testing.columns.isin(self.exclude_cols)].to_numpy()
        labels_test = self.testing['Label'].to_numpy()
        
        y_pred = []
        
        for i in range(features_test.shape[0]):
            y_pred.append(self.get_pred(features_test[i]))
            
        idx = np.arange(0, len(y_pred), dtype=int)
        accuracy_count = len(idx[np.array(y_pred)[idx] == labels_test[idx]])
        accuracy = accuracy_count / len(labels_test)

        print('Labels: ', labels_test)
        print('Predictions: ', y_pred)
        print('Accuracy: ', accuracy)