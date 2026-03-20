from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, classification_report
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self, model):
        self.model = model
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:\n", report)
        return acc, report

    def cross_validate(self, X, y, cv=10):
        """
        Perform k-fold
        """
        # Trains the models as well
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        print(f"{cv}-fold cross validation Accuracy: {np.mean(scores):.4f} with an std of {np.std(scores):.4f}")
        return scores
    
class RandomForestClassifierWrapper(Classifier):
    def __init__(self, n_trees=100, max_depth=None, random_state=42):
        model = RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            random_state=random_state
        )
        super().__init__(model)

    def surrogate_tree(self, X, feature_names=None, class_names=None, max_depth=5):
        """
        Create surrogate tree for random forest
        """
        #surrogate tree is built by fitting DT to predictions from rf on set of data
        rf_preds = self.model.predict(X)
        surrogate_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        surrogate_tree.fit(X, rf_preds)
        
        #plot tree
        plt.figure(figsize=(20,10))
        plot_tree(
            surrogate_tree,
            feature_names=feature_names,
            class_names=[str(c) for c in np.unique(rf_preds)] if class_names is None else class_names,
            filled=True,
            fontsize=10
        )
        plt.show()
        
        #return surrogate tre
        return surrogate_tree

class LogisticRegressionWrapper(Classifier):
    def __init__(self, max_iter=500, random_state=42):
        model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            n_jobs=-1
        )
        super().__init__(model)

class NaiveBayesClassifierWrapper(Classifier):
    def __init__(self):
        model = GaussianNB()
        super().__init__(model)