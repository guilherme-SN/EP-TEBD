import os
import logging
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50

# Configura o Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingExtractor:
    # Por padrão, utiliza o modelo ResNet50 para extrair os embeddings
    def __init__(self, model_path: str = "models/resnet50.h5"):
        try:
            if os.path.exists(model_path):
                logger.info("Carregando modelo ResNet50")
                self.model = tf.keras.models.load_model(model_path)
            else:
                logger.info("Criando modelo ResNet50")
                base_model = ResNet50(weights='imagenet',   # Utiliza os pesos pré-treinados com o dataset ImageNet
                                      include_top=False,    # Remove a camada de classificação do modelo
                                      pooling='avg')        # Define o pooling para pegar a média das características

                input_tensor = tf.keras.Input(shape=(224, 224, 3))  # Espera imagens RBG de 224x224 pixels
                features_2048_dimension = base_model(input_tensor, training=False)

                # Camada Dense para reduzir as dimensões para de 2048 para 128
                embedding_128_dimension = tf.keras.layers.Dense(128, activation='relu')(features_2048_dimension)

                # Cria o modelo
                self.model = tf.keras.Model(inputs=input_tensor, outputs=embedding_128_dimension)
                self.model.save(model_path)
                logger.info("Modelo ResNet50 criado")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            raise e

    def extract_embedding(self, image_path: str) -> list[float]:
        # Carrega a imagem e já redimensiona para 224x224
        img = image.load_img(image_path, target_size=(224, 224))

        img_array = image.img_to_array(img)             # Converte para array
        img_array = np.expand_dims(img_array, axis=0)   # Configura o batch_size para 1, pois modelo espera (1, 224, 224, 3)
        img_array = preprocess_input(img_array)         # Pré-processa a imagem para normalizar

        # Extrai o embedding usando o modelo
        extracted_embedding = self.model.predict(img_array)
        return extracted_embedding[0].tolist()
