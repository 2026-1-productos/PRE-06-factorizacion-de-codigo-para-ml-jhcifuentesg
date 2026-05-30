#
# Busque los mejores parametros de un modelo ElasticNet para predecir
# la calidad del vino usando el dataset de calidad del vino tinto de UCI.
#
# Consideere los siguentes valores de los hiperparametros y obtenga el
# mejor modelo.
# (alpha, l1_ratio):
#    (0.5, 0.5), (0.2, 0.2), (0.1, 0.1), (0.1, 0.05), (0.3, 0.2)
#

# importacion de librerias
from sklearn.linear_model import ElasticNet

from homework.src._internals.calculate_metrics import calculate_metrics
from homework.src._internals.experimetnt import Run_experiment
from homework.src._internals.prepare_data import prepare_data
from homework.src._internals.Print_Metrics import Print_Metrics
from homework.src._internals.save_model import save_model

Run_experiment(estimator=ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=12345))
