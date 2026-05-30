from sklearn.neighbors import KNeighborsRegressor

from .calculate_metrics import calculate_metrics
from .prepare_data import prepare_data
from .Print_Metrics import Print_Metrics
from .save_model import save_model


def Run_experiment(estimator):
    x_train, x_test, y_train, y_test = prepare_data()

    estimator.fit(x_train, y_train)
    save_model(estimator)

    print()
    print(estimator, ":", sep="")

    mse, mae, r2 = calculate_metrics(x_train, y_train, estimator)
    Print_Metrics(mse, mae, r2, Title="Metricas de entrenamiento:")

    mse, mae, r2 = calculate_metrics(x_test, y_test, estimator)
    Print_Metrics(mse, mae, r2, Title="Metricas de testing:")
