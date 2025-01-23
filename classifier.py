import pandas as pd
import time
import os
import csv
import warnings

from abc import ABC, abstractmethod
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree  import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings(action='ignore') #some classifiers have some issues


class Classifier(ABC):
    def __init__(self, data):
        self.data = data

        self.model_name = None
        self.model_abb = None

        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

    def _prep_split_data(self):
        """Function which prepares and splits data into training and test set."""
        attack_LE = LabelEncoder()
        self.data['attack'] = attack_LE.fit_transform(self.data['attack'])

        X = self.data.drop(['attack', 'level', 'attack_state'], axis=1)
        Y = self.data['attack_state']

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size= 0.25)

        Ro_scaler = RobustScaler()
        self.X_train = Ro_scaler.fit_transform(self.X_train)
        self.X_test = Ro_scaler.transform(self.X_test)

    def _train_and_save(self):
        """Train the model and save the evaluation results."""
        start_time = time.time()
        self.model_abb = self._initialize_model()
        self.model_abb.fit(self.X_train, self.Y_train)
        evaluation = self._evaluate(self.model_name, self.model_abb, start_time)
        self._save_result(evaluation)

    def _evaluate(self, model_name, model_abb, start_time):
        """Function which evaluates how classifier works in terms of given metrics."""
        pred_value = model_abb.predict(self.X_test)

        #basic metrics
        accuracy = metrics.accuracy_score(self.Y_test, pred_value) 
        sensitivity = metrics.recall_score(self.Y_test, pred_value) 
        precision = metrics.precision_score(self.Y_test, pred_value) 
        f1_score = metrics.f1_score(self.Y_test, pred_value)
        specificity = metrics.recall_score(self.Y_test, pred_value, pos_label=0) 
        balanced_accuracy = metrics.balanced_accuracy_score(self.Y_test, pred_value)
        mcc = metrics.matthews_corrcoef(self.Y_test, pred_value)

        #confusion matrix
        cm = metrics.confusion_matrix(self.Y_test, pred_value)
        TN, FP, FN, TP = cm.ravel()
        fpr = FP / (FP + TN) if (FP + TN) > 0 else 0 
        fnr = FN / (FN + TP) if (FN + TP) > 0 else 0

        #calculating time for training and calculating metrics
        end_time = time.time()
        elapsed_time = end_time - start_time

        evaluation_result = {
            "Model name" : model_name, #maximize
            "Accuracy" : accuracy, #maximize
            "Sensitivity" : sensitivity, #maximize
            "Precision" : precision, #maximize
            "F1 Score" : f1_score, #maximize
            "Specificity" : specificity, #maximize
            "Balanced accuracy" : balanced_accuracy, #maximize
            "MCC" : mcc, #maximize
            "False Positive Rate" : fpr, #minimized
            "False Negative Rate" : fnr, #minimized
            "Time" : elapsed_time #minimized
        }
        
        return evaluation_result
    
    def _save_result(self, evaluation_result, file_name='evaluation_results.csv'):
        """Saves evaluation results to a CSV file, creating the file if it doesn't exist."""
        file_exists = os.path.isfile(file_name)
        
        with open(file_name, mode='a', newline='', encoding='utf-8') as file:
            fieldnames = evaluation_result.keys()
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
            writer.writerow(evaluation_result)

    @abstractmethod
    def _initialize_model(self):
        """Abstract method to initialize the model. Must be overridden in subclass."""
        pass


class Logistic(Classifier):
    def __init__(self, data):
        super().__init__(data)
        self.model_name = 'Logistic Regression'

    def _initialize_model(self):
        return LogisticRegression()


class DecistionTree(Classifier): #TODO check paramaters of DT classifier
    def __init__(self, data):
        super().__init__(data)
        self.model_name = 'Decision Tree'

    def _initialize_model(self):
        return DecisionTreeClassifier(max_features=6, max_depth=4) #not sure about these parameters
    

class RandomForest(Classifier):
    def __init__(self, data):
        super().__init__(data)
        self.model_name = 'Random Forest'

    def _initialize_model(self):
        return RandomForestClassifier()
    

class KNN(Classifier): #TODO check paramaters of KNN classifier
    def __init__(self, data):
        super().__init__(data)
        self.model_name = 'KNearestNeighbors'

    def _initialize_model(self):
        return KNeighborsClassifier(n_neighbors=6) #not sure about these parameters
    

class NaiveBayes(Classifier):
    def __init__(self, data):
        super().__init__(data)
        self.model_name = 'NaiveBayes'

    def _initialize_model(self):
        return GaussianNB()


class GradientBoost(Classifier):
    def __init__(self, data):
        super().__init__(data)
        self.model_name = 'GradientBoosting'

    def _initialize_model(self):
        return GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1) #not sure about these parameters


class MLP(Classifier):
    def __init__(self, data):
        super().__init__(data)
        self.model_name = 'MLP'

    def _initialize_model(self):
        return MLPClassifier(max_iter=300)
    

class XGBC(Classifier):
    def __init__(self, data):
        super().__init__(data)
        self.model_name = 'XGBC'

    def _initialize_model(self):
        return XGBClassifier()
    

class LGBMC(Classifier):
    def __init__(self, data):
        super().__init__(data)
        self.model_name = 'LGBMC'

    def _initialize_model(self):
        return LGBMClassifier(verbose=-1) #supressing logs from here
    

class ClassifierFactory:
    @staticmethod
    def create_classifier(name, data):
        classifiers = {
            "logistic": Logistic,
            "decision_tree": DecistionTree,
            "random_forest" : RandomForest,
            "knn" : KNN,
            "naive_bayes" : NaiveBayes,
            "gradient_boost" : GradientBoost,
            "mlp" : MLP,
            "xgbc" : XGBC,
            "lgbmc" : LGBMC,
        }
        if name not in classifiers:
            raise ValueError(f"Classifier '{name}' is not defined.")
        return classifiers[name](data)


def _train_all(num_iteration):
    """Training and saving data to csv file"""
    train_data = pd.read_csv("data/train_prepared.csv")

    classifiers = ["logistic", "decision_tree", "random_forest", "knn", "naive_bayes", 
                   "gradient_boost", "mlp", "xgbc", "lgbmc"]
    
    models = [ClassifierFactory.create_classifier(name, train_data) for name in classifiers]

    for _ in range(num_iteration):
        for model in models:
            model._prep_split_data()
            model._train_and_save()

NUM_ITER = 2
_train_all(NUM_ITER) #comment this line to prevent further learning processes