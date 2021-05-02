import pandas as pd
from tabulate import tabulate
from preprocessing import DataProcessor, get_main_variables

class DataFrameReporter(object):
    '''
    Helper class that reports nulls and datatypes of train and test data
    '''
    def __init__(self, X_train, X_test, target_column):
        '''
        Constructor for the class.
        Needs train and test data and also the target column in train.
        '''
        self.X_train = X_train
        self.X_test = X_test
        self.target_column = target_column
        
    def analyze_X(self, X):
        '''
        Analyses the DataFrame you pass and returns a report of nulls, distribution and other goodies.
        '''
        
        if self.target_column in X.columns:
            X = X.drop(self.target_column, axis = 1)
            
        dtypes = X.dtypes.to_frame().rename(columns = {0:"Dtypes"})

        nulls_in_train = X.isnull().sum().to_frame().rename(columns = {0:"Absolute_nulls"})
        nulls_in_train["Relative_nulls"] = nulls_in_train["Absolute_nulls"]/X.shape[0]
        nulls_in_train["Relative_nulls"] = nulls_in_train["Relative_nulls"].apply(lambda number: round(number, 3) * 100)
        nulls_in_train = pd.concat([nulls_in_train, dtypes], axis = 1)
        nulls_in_train["Shape"] = X.shape[0]
        nulls_in_train = nulls_in_train[["Dtypes", "Shape", "Absolute_nulls", "Relative_nulls"]]

        describe_values_num = X.describe().T
        report_df = pd.concat([nulls_in_train, describe_values_num], axis = 1)

        object_columns = X.select_dtypes("object").columns
        unique_categories = {col:X[col].nunique() for col in object_columns}
        unique_cat_df = pd.DataFrame(data = unique_categories.values(), index = unique_categories.keys(), columns = ["Unique_category"])

        report_df = pd.concat([report_df, unique_cat_df], axis = 1)

        report_df.fillna("", inplace = True)
        report_df.sort_values("Dtypes", ascending = True, inplace = True)
        
        return report_df
        
    def get_reports(self):
        '''
        Calls analyze_X method and returns report DataFrame for train and test.
        '''
        report_train = self.analyze_X(X = self.X_train)
        report_test = self.analyze_X(X = self.X_test)
        
        return report_train, report_test

def main():
    PATH_TRAIN, PATH_TEST, ID_COLUMN, TRAIN_COLUMNS, TARGET_COLUMN = get_main_variables()
    dp = DataProcessor(PATH_TRAIN, PATH_TEST, ID_COLUMN, TRAIN_COLUMNS, TARGET_COLUMN)

    X_train, y, X_test = dp.process_data()
    X_train[TARGET_COLUMN] = y

    df_rep = DataFrameReporter(X_train, X_test, TARGET_COLUMN)
    report_train, report_test = df_rep.get_reports()

    print(tabulate(report_train))
    print(tabulate(report_test))

if __name__ == "__main__":
    main()