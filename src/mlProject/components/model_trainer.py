import pandas as pd
import os
from mlProject import logger
from sklearn.linear_model import ElasticNet
import joblib
from mlProject.entity.config_entity import (ModelTrainerConfig)


# Crear una clase para entrenar el modelo
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    def train(self):
        # Cargar el dataset
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # establer la variable "X" y "y" tanto en el dataset de entrenamient y test
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        # Crear el modelo 
        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        # Entrenar el modelo
        lr.fit(train_x, train_y)
        # Guardar el modelo entrenado en la ruta establecida(dentro de artifacts)
        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))
