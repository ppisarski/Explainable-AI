import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from definitions import ROOT_DIR
import lime

DATA_URL = os.path.join(ROOT_DIR, 'input/winequality-white.csv')


def main():
    wine = pd.read_csv(DATA_URL)

    x = wine.drop('quality', axis=1)
    y = wine['quality']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Model training
    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)

    # Model interpretation
    # Create tabular explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(x_train),
        feature_names=x_train.columns,
        class_names=['bad', 'good'],
        mode='classification'
    )


if __name__ == "__main__":
    main()
