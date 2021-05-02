import os
import pandas as pd

class DataLoader():
    def __init__(self, path_train, path_test, id_column):
        self.path_train = path_train
        self.path_test = path_test
        self.id_column = id_column

    def load_data(self):
        if os.path.splitext(self.path_train)[1] == ".csv":
            X_train = pd.read_csv(self.path_train)
            X_train.set_index(self.id_column, inplace = True)

        if os.path.splitext(self.path_test)[1] == ".csv":
            X_test = pd.read_csv(self.path_test)
            X_test.set_index(self.id_column, inplace = True)
    
        return X_train, X_test

class DataProcessor(DataLoader):
    def __init__(self, path_train, path_test, id_column, train_columns, target_column):
        super().__init__(path_train, path_test, id_column)
        self.train_columns = train_columns
        self.target_column = target_column

    def process_data(self):
        X_train, X_test = self.load_data()

        for col in self.train_columns:
            assert_message = f"{col} not found in train or test df"
            assert col in X_train.columns and col in X_test.columns, assert_message

        y = X_train[self.target_column]
        X_train = X_train[self.train_columns]

        X_test = X_test[self.train_columns]

        return X_train, y, X_test

def get_main_variables():

    PATH_DATA = os.path.join(os.getcwd(), "data")
    PATH_TRAIN = os.path.join(PATH_DATA, "train.csv")
    PATH_TEST = os.path.join(PATH_DATA, "test.csv")

    ID_COLUMN = "PassengerId"

    TRAIN_COLUMNS = ["Age", "Sex", "Fare", "Embarked"]
    
    TARGET_COLUMN = "Survived"

    return PATH_TRAIN, PATH_TEST, ID_COLUMN, TRAIN_COLUMNS, TARGET_COLUMN

def main():

    PATH_TRAIN, PATH_TEST, ID_COLUMN, TRAIN_COLUMNS, TARGET_COLUMN = get_main_variables()
    dp = DataProcessor(PATH_TRAIN, PATH_TEST, ID_COLUMN, TRAIN_COLUMNS, TARGET_COLUMN)

    X_train, y, X_test = dp.process_data()
    print(X_train.columns)

if __name__ == "__main__":
    main()