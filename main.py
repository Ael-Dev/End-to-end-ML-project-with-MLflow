"""
Ejecutar las diferentes etapas de un pipeline de entrenamiento 
de un modelo de aprendizaje automático. Estas etapas son:
    - Ingestión de datos: se encarga de obtener los datos desde 
      una fuente externa y guardarlos en un formato adecuado.
    - Validación de datos: se encarga de verificar la calidad y 
      la integridad de los datos, y detectar posibles anomalías o errores.
    - Transformación de datos: se encarga de aplicar las operaciones necesarias para limpiar, 
      normalizar, escalar, codificar y dividir los datos en conjuntos de entrenamiento y prueba.
    - Entrenamiento del modelo: se encarga de definir, compilar, ajustar y entrenar el modelo 
      usando los datos de entrenamiento.
    - Evaluación del modelo: se encarga de medir el rendimiento del modelo 
      usando los datos de prueba y diferentes métricas.

Para cada etapa, se importa una clase que implementa el pipeline correspondiente desde el paquete mlProject
"""

from mlProject import logger
from mlProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e

# -------------------------------------------------------------------------