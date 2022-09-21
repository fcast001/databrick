# Databricks notebook source
import numpy as np
import pandas as pd
 
# In the following lines, replace <username> with your username.
white_wine = pd.read_csv("/dbfs/FileStore/tables/winequality_white.csv", sep=';')
red_wine = pd.read_csv("/dbfs/FileStore/tables/winequality_red.csv", sep=';')

# COMMAND ----------

red_wine['is_red'] = 1
white_wine['is_red'] = 0
 
data = pd.concat([red_wine, white_wine], axis=0)
# Remove spaces from column names
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)


# COMMAND ----------

data

# COMMAND ----------

import seaborn as sns
sns.distplot(data.quality, kde=False)

# COMMAND ----------

# Parece que los puntajes de calidad normalmente se distribuyen entre 3 y 9.
# Definir un vino como de alta calidad si tiene una calidad >= 7.

# COMMAND ----------

high_quality = (data.quality >= 7).astype(int)
data.quality = high_quality

# COMMAND ----------

data

# COMMAND ----------

import matplotlib.pyplot as plt
 
dims = (3, 4)
 
f, axes = plt.subplots(dims[0], dims[1], figsize=(25, 15))
axis_i, axis_j = 0, 0
for col in data.columns:
  if col == 'is_red' or col == 'quality':
    continue # Box plots cannot be used on indicator variables
  sns.boxplot(x=high_quality, y=data[col], ax=axes[axis_i, axis_j])
  axis_j += 1
  if axis_j == dims[1]:
    axis_i += 1
    axis_j = 0
    

# COMMAND ----------

# MAGIC %md
# MAGIC En los diagramas de caja anteriores, algunas variables se destacan como buenos predictores univariados de calidad.
# MAGIC 
# MAGIC En el diagrama de caja de alcohol, el contenido de alcohol medio de los vinos de alta calidad es mayor incluso que el cuantil 75 de los vinos de baja calidad. El alto contenido de alcohol se correlaciona con la calidad.
# MAGIC En el diagrama de caja de densidad, los vinos de baja calidad tienen una mayor densidad que los vinos de alta calidad. La densidad está inversamente correlacionada con la calidad.

# COMMAND ----------

# MAGIC %md
# MAGIC Preprocesar datos
# MAGIC Antes de entrenar un modelo, compruebe si faltan valores y divida los datos en conjuntos de entrenamiento y validación.

# COMMAND ----------

data.isna().any()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preparar el conjunto de datos para el modelo de referencia de entrenamiento
# MAGIC Divida los datos de entrada en 3 conjuntos:
# MAGIC 
# MAGIC * Entrenar (60 % del conjunto de datos utilizado para entrenar el modelo)
# MAGIC * Validación (20 % del conjunto de datos utilizado para ajustar los hiperparámetros)
# MAGIC * Prueba (20 % del conjunto de datos utilizado para informar el rendimiento real del modelo en un conjunto de datos no visto)

# COMMAND ----------

from sklearn.model_selection import train_test_split
 
X = data.drop(["quality"], axis=1)
y = data.quality
 
# Dividir los datos de entrenamiento
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, random_state=123)
 
# Divida los datos restantes por igual en validación y prueba
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, random_state=123)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Construir un modelo de línea de base
# MAGIC Esta tarea parece adecuada para un clasificador de bosque aleatorio, ya que la salida es binaria y puede haber interacciones entre múltiples variables.
# MAGIC 
# MAGIC El siguiente código construye un clasificador simple usando scikit-learn. Utiliza MLflow para realizar un seguimiento de la precisión del modelo y para guardar el modelo para su uso posterior.

# COMMAND ----------

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle
import time
 
# El método de predicción de RandomForestClassifier de sklearn devuelve una clasificación binaria (0 o 1).
# El siguiente código crea una función contenedora, SklearnModelWrapper, que usa
# el método predict_proba para devolver la probabilidad de que la observación pertenezca a cada clase.
 
class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]
 
# mlflow.start_run crea una nueva ejecución de MLflow para realizar un seguimiento del rendimiento de este modelo.
# Dentro del contexto, llame a mlflow.log_param para realizar un seguimiento de los parámetros utilizados, y
# mlflow.log_metric para registrar métricas como la precisión.
with mlflow.start_run(run_name='untuned_random_forest'):
  n_estimators = 10
  model = RandomForestClassifier(n_estimators=n_estimators, random_state=np.random.RandomState(123))
  model.fit(X_train, y_train)
 
  # predict_proba devuelve [prob_negative, prob_positive], por lo tanto, divida la salida con [:, 1]
  predictions_test = model.predict_proba(X_test)[:,1]
  auc_score = roc_auc_score(y_test, predictions_test)
  mlflow.log_param('n_estimators', n_estimators)
  # Use el área bajo la curva ROC como métrica.
  mlflow.log_metric('auc', auc_score)
  wrappedModel = SklearnModelWrapper(model)
  # Registre el modelo con una firma que defina el esquema de las entradas y salidas del modelo.
  # Cuando se implemente el modelo, esta firma se utilizará para validar las entradas.
  signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
  
  # MLflow contiene utilidades para crear un entorno de conda utilizado para servir modelos.
  # Las dependencias necesarias se agregan a un archivo conda.yaml que se registra junto con el modelo.
  conda_env =  _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
        additional_conda_channels=None,
    )
  mlflow.pyfunc.log_model("random_forest_model", python_model=wrappedModel, conda_env=conda_env, signature=signature)

# COMMAND ----------

# MAGIC %md
# MAGIC Examine la salida de importancia de las características aprendidas por el modelo como una verificación de cordura.

# COMMAND ----------

feature_importances = pd.DataFrame(model.feature_importances_, index=X_train.columns.tolist(), columns=['importance'])
feature_importances.sort_values('importance', ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Como se ilustra en los diagramas de caja mostrados anteriormente, tanto el alcohol como la densidad son importantes para predecir la calidad.
# MAGIC 
# MAGIC Ha registrado el área bajo la curva ROC (AUC) en MLflow. Haga clic en Experimento en la parte superior derecha para mostrar la barra lateral Ejecuciones de experimentos.
# MAGIC 
# MAGIC El modelo logró un AUC de 0,854.
# MAGIC 
# MAGIC Un clasificador aleatorio tendría un AUC de 0,5, y los valores de AUC más altos son mejores. Para obtener más información, consulte Curva característica de funcionamiento del receptor

# COMMAND ----------

# MAGIC %md
# MAGIC Registre el modelo en el registro de modelos de MLflow
# MAGIC Al registrar este modelo en Model Registry, puede hacer referencia fácilmente al modelo desde cualquier lugar dentro de Databricks.
# MAGIC 
# MAGIC La siguiente sección muestra cómo hacer esto mediante programación, pero también puede registrar un modelo mediante la interfaz de usuario. Consulte "Crear o registrar un modelo mediante la interfaz de usuario" ( AWS | Azure | GCP ).

# COMMAND ----------

run_id = mlflow.search_runs(filter_string='tags.mlflow.runName = "untuned_random_forest"').iloc[0].run_id

# COMMAND ----------

# Si ve el error "PERMISSION_DENIED: El usuario no tiene ningún nivel de permiso asignado al modelo registrado",
# la causa puede ser que ya exista un modelo con el nombre "wine_quality". Intente usar un nombre diferente.
model_name = "wine_quality"
model_version = mlflow.register_model(f"runs:/{run_id}/random_forest_model", model_name)
 
# Registrar el modelo lleva unos segundos, así que agregue un pequeño retraso
time.sleep(15)


# COMMAND ----------

# MAGIC %md
# MAGIC Ahora debería ver el modelo en la página Modelos. Para mostrar la página Modelos, haga clic en el icono Modelos en la barra lateral izquierda.
# MAGIC 
# MAGIC A continuación, transfiera este modelo a producción y cárguelo en este cuaderno desde Model Registry.

# COMMAND ----------

from mlflow.tracking import MlflowClient
 
client = MlflowClient()
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Production",
)

# COMMAND ----------

# MAGIC %md
# MAGIC La página Modelos ahora muestra la versión del modelo en la etapa "Producción".
# MAGIC 
# MAGIC Ahora puede consultar el modelo utilizando la ruta "modelos:/calidad_del_vino/producción".

# COMMAND ----------

model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
 
# Sanity-check: This should match the AUC logged by MLflow
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experimentar con un nuevo modelo
# MAGIC El modelo de bosque aleatorio funcionó bien incluso sin ajuste de hiperparámetros.
# MAGIC 
# MAGIC El siguiente código usa la biblioteca xgboost para entrenar un modelo más preciso. Ejecuta un barrido paralelo de hiperparámetros para entrenar varios modelos en paralelo, utilizando Hyperopt y SparkTrials. Como antes, el código realiza un seguimiento del rendimiento de cada configuración de parámetros con MLflow.

# COMMAND ----------

from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from math import exp
import mlflow.xgboost
import numpy as np
import xgboost as xgb
 
search_space = {
  'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
  'learning_rate': hp.loguniform('learning_rate', -3, 0),
  'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
  'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
  'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
  'objective': 'binary:logistic',
  'seed': 123, # Establecer una semilla para el entrenamiento determinista  
}
 
def train_model(params):
  # Con el registro automático de MLflow, los hiperparámetros y el modelo entrenado se registran automáticamente en MLflow.
  mlflow.xgboost.autolog()
  with mlflow.start_run(nested=True):
    train = xgb.DMatrix(data=X_train, label=y_train)
    validation = xgb.DMatrix(data=X_val, label=y_val)
    # Pase el conjunto de validación para que xgb pueda rastrear una métrica de evaluación. XGBoost finaliza el   
    # entrenamiento cuando la métrica  de evaluación
    # ya no mejora.
    booster = xgb.train(params=params, dtrain=train, num_boost_round=1000,\
                        evals=[(validation, "validation")], early_stopping_rounds=50)
    validation_predictions = booster.predict(validation)
    auc_score = roc_auc_score(y_val, validation_predictions)
    mlflow.log_metric('auc', auc_score)
 
    signature = infer_signature(X_train, booster.predict(train))
    mlflow.xgboost.log_model(booster, "model", signature=signature)
    
    # Establecer la pérdida en -1*auc_score para que fmin maximice el auc_score
    return {'status': STATUS_OK, 'loss': -1*auc_score, 'booster': booster.attributes()}
 
# Un mayor paralelismo conducirá a aceleraciones, pero un barrido de hiperparámetros menos óptimo.
# Un valor razonable para el paralelismo es la raíz cuadrada de max_evals.
spark_trials = SparkTrials(parallelism=10)
 
# Ejecute fmin dentro de un contexto de ejecución de MLflow para que cada configuración de hiperparámetro se registre como una ejecución secundaria de un padre
# ejecutar llamado "xgboost_models" .
with mlflow.start_run(run_name='xgboost_models'):
  best_params = fmin(
    fn=train_model, 
    space=search_space, 
    algo=tpe.suggest, 
    max_evals=96,
    trials=spark_trials,
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Usó MLflow para registrar el modelo producido por cada configuración de hiperparámetro. El siguiente código busca la ejecución con mejor rendimiento y guarda el modelo en Model Registry.

# COMMAND ----------

best_run = mlflow.search_runs(order_by=['metrics.auc DESC']).iloc[0]
print(f'AUC of Best Run: {best_run["metrics.auc"]}')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Actualizar el modelo de producción wine_qualityen el registro de modelos de MLflow
# MAGIC Anteriormente, guardó el modelo de línea base en Model Registry con el nombre wine_quality. Ahora que ha creado un modelo más preciso, actualice wine_quality.

# COMMAND ----------

new_model_version = mlflow.register_model(f"runs:/{best_run.run_id}/model", model_name)
time.sleep(15)

# COMMAND ----------

# MAGIC %md
# MAGIC Haga clic en Modelos en la barra lateral izquierda para ver que el wine_qualitymodelo ahora tiene dos versiones.
# MAGIC 
# MAGIC El siguiente código promociona la nueva versión a producción.

# COMMAND ----------

# Archivar la versión del modelo anterior
client.transition_model_version_stage(
  name=model_name,
  version=model_version.version,
  stage="Archived"
)
 
# Promocionar la nueva versión del modelo a Producción
client.transition_model_version_stage(
  name=model_name,
  version=new_model_version.version,
  stage="Production"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Los clientes que llaman a load_model ahora reciben el nuevo modelo.

# COMMAND ----------

# Este código es el mismo que el último bloque de "Construir un modelo de referencia". ¡No se requiere ningún cambio para que los clientes obtengan el nuevo modelo!
model = mlflow.pyfunc.load_model(f"models:/{model_name}/production")
print(f'AUC: {roc_auc_score(y_test, model.predict(X_test))}')

# COMMAND ----------

# MAGIC %md
# MAGIC El valor auc en el conjunto de prueba para el nuevo modelo es 0,90. ¡Superaste la línea de base!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inferencia por lotes
# MAGIC Hay muchos escenarios en los que es posible que desee evaluar un modelo en un corpus de datos nuevos. Por ejemplo, es posible que tenga un lote de datos nuevo o que necesite comparar el rendimiento de dos modelos en el mismo corpus de datos.
# MAGIC 
# MAGIC El siguiente código evalúa el modelo en datos almacenados en una tabla Delta, usando Spark para ejecutar el cálculo en paralelo.

# COMMAND ----------

# Para simular un nuevo corpus de datos, guarde los datos existentes de X_train en una tabla Delta.
# En el mundo real, este sería un nuevo lote de datos.
spark_df = spark.createDataFrame(X_train)
# Reemplace <nombre de usuario> con su nombre de usuario antes de ejecutar esta celda.
table_path = "dbfs:/felipe.castillo@axity.com/delta/wine_data"
# Eliminar el contenido de esta ruta en caso de que esta celda ya se haya ejecutado
dbutils.fs.rm(table_path, True)
spark_df.write.format("delta").save(table_path)

# COMMAND ----------


