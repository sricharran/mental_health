import os
import sys
from dataclasses import dataclass

# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

#Bagging
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models
from sklearn.metrics import accuracy_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)

            models = {
                "Random Forest": RandomForestClassifier(n_estimators = 20),
                "Decision Tree": DecisionTreeClassifier(max_depth=3, min_samples_split=8, max_features=6, criterion='entropy', min_samples_leaf=7),
                "Extra Tree": ExtraTreesClassifier(n_estimators=250,random_state=0),
                "Logistic Regression": LogisticRegression(),
                "Bagging": BaggingClassifier(DecisionTreeClassifier(), max_samples=1.0, max_features=1.0, bootstrap_features=False),
                "KNN Classifier": KNeighborsClassifier(n_neighbors=5),
                "AdaBoost Classifier": AdaBoostClassifier(estimator=clf, n_estimators=500),
            }
            params = {
                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ['sqrt', 'log2', None]
                },

                "Decision Tree": {
                    "criterion": ['gini', 'entropy'],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ['sqrt', 'log2', None]
                },

                "Extra Tree": {
                    "n_estimators": [100, 250, 500],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ['sqrt', 'log2', None]
                },

                "Logistic Regression": {
                    "penalty": ['l1', 'l2'],
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ['saga'],
                    "max_iter": [500]
                },

                "Bagging": {
                    "n_estimators": [10, 50, 100],
                    "max_samples": [0.5, 0.7, 1.0],
                    "max_features": [0.5, 0.7, 1.0],
                    "bootstrap": [True, False],
                    "bootstrap_features": [True, False]
                },

                "KNN Classifier": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ['uniform', 'distance'],
                    "metric": ['euclidean', 'manhattan', 'minkowski']
                },

                "AdaBoost Classifier": {
                    "n_estimators": [50, 100, 300, 500],
                    "learning_rate": [0.01, 0.1, 1],
                    "estimator__max_depth": [1, 2, 3]
                }
            }



            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,
                                              X_test=X_test,y_test=y_test,
                                              models=models,param=params)
            
            ## To get best model score from dict
            best_model_name = max(model_report, key=lambda x: model_report[x]["score"])
            best_model_score = model_report[best_model_name]["score"]
            best_model = model_report[best_model_name]["model"]
            

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predicted)
            return accuracy        
            
        except Exception as e:
            raise CustomException(e,sys)