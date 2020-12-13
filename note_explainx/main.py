import pandas as pd
from explainx import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA_URL = 'input/winequality-white.csv'


def load_data(filepath=DATA_URL):
    df = pd.read_csv(filepath, sep=';')
    x = df.drop('quality', axis=1)
    y = df['quality']
    return x, y


def main():
    # x_data, y_data = load_data()
    x_data, y_data = explainx.dataset_iris()
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

    # Train a RandomForest Model
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    explainx.ai(x_test, y_test, model, model_name="randomforest")


if __name__ == "__main__":
    main()
