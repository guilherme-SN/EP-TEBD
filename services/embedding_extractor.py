import os
import logging
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Configura o Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingExtractor:
    def __init__(
            self,
            embedding_model_path: str = "models/libras_embeddings_model2.h5",
            classifier_model_path: str = "models/libras_model2.h5",
            train_data_dir: str = "../data/train/"
    ):
        self.embedding_model_path = embedding_model_path
        self.classifier_model_path = classifier_model_path
        self.train_data_dir = train_data_dir

        # Se o modelo de embeddings não existir, treina e exporta
        if not os.path.exists(self.embedding_model_path):
            logger.info("Modelo de embeddings não encontrado. Iniciando treinamento e exportação...")
            self.train_and_export_model()

        # Carrega o modelo de embeddings
        logger.info("Carregando modelo de embeddings")
        self.model = tf.keras.models.load_model(self.embedding_model_path)
        self.model.compile(optimizer='adam', loss='mse')

        @tf.function(reduce_retracing=True)
        def predict_fn(x):
            return self.model(x, training=False)

        self.predict_fn = predict_fn

    def train_and_export_model(self):
        # Gera dados
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.2,
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=False
        )

        train_gen = datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        val_gen = datagen.flow_from_directory(
            self.train_data_dir,
            target_size=(224, 224),
            batch_size=32,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        # Cria o modelo do zero
        base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)  # Embedding final
        output = Dense(train_gen.num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Treinamento
        model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=[early_stop])

        # Salva modelo de classificação
        os.makedirs(os.path.dirname(self.classifier_model_path), exist_ok=True)
        model.save(self.classifier_model_path)
        logger.info(f"Modelo de classificação salvo em {self.classifier_model_path}")

        # Exporta o modelo de embeddings
        embedding_output = model.layers[-2].output
        embedding_model = tf.keras.Model(inputs=model.input, outputs=embedding_output)
        embedding_model.save(self.embedding_model_path)
        logger.info(f"Modelo de embeddings salvo em {self.embedding_model_path}")

    def extract_embedding(self, image_path: str) -> list[float]:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        extracted_embedding = self.predict_fn(img_array)
        return extracted_embedding.numpy()[0].tolist()

    def extract_embeddings_batch(self, image_paths: list[str], batch_size: int = 32) -> list[list[float]]:
        all_embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_arrays = []

            for img_path in batch_paths:
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                batch_arrays.append(img_array[0])

            batch_arrays = np.array(batch_arrays)
            batch_embeddings = self.predict_fn(batch_arrays)
            all_embeddings.extend(batch_embeddings.numpy().tolist())

        return all_embeddings