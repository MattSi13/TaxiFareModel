# imports
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data, clean_data


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                            ('stdscaler', StandardScaler())])
        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                            ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc_pipe = ColumnTransformer([('distance', dist_pipe, [
            "pickup_latitude", "pickup_longitude", 'dropoff_latitude',
            'dropoff_longitude'
        ]), ('time', time_pipe, ['pickup_datetime'])],
                                        remainder="drop")
        pipe = Pipeline([('preproc', preproc_pipe),
                        ('linear_model', LinearRegression())])
        self.pipeline = pipe
        return self

    def run(self):
        """set and train the pipeline"""
        self.pipeline.fit(self.X_train, self.y_train)
        return self.pipeline

    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(self.X_test)
        rmse = compute_rmse(y_pred, self.y_test)
        return rmse


if __name__ == "__main__":
    # get data
    df = get_data()
    # clean data
    df_clean = clean_data(df)
    # set X and y
    y = df_clean["fare_amount"]
    X = df_clean.drop("fare_amount", axis=1)
    trainer = Trainer(X, y)
    # pipeline
    trainer.set_pipeline()
    # train
    trainer.run()
    # evaluate
    rmse = trainer.evaluate()
    print('TODO')
