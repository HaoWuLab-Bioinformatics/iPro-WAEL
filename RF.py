from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
def pred(x_train, y_train, x_weight, x_test, n_trees):
    model = RandomForestClassifier(n_estimators=n_trees, random_state=10)
    model.fit(x_train, y_train)
    joblib.dump(model, 'RF.model')
    weight_proba = model.predict_proba(x_weight)[:, 1]  # probability
    test_proba = model.predict_proba(x_test)[:, 1]  # probability
    test_class = model.predict(x_test)  # class
    return weight_proba, test_proba, test_class
