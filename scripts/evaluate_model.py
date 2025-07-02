import os
import sys
import logging
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# Configura o Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Muda para o diretório raiz
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

TEST_DIR = os.path.join(PROJECT_ROOT, "data/test/")
CLASSIFIER_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/libras_model2.h5")

if __name__ == "__main__":
    logger.info(f"Carregando o classificador: {CLASSIFIER_MODEL_PATH}")
    try:
        model = tf.keras.models.load_model(CLASSIFIER_MODEL_PATH)
        logger.info("Modelo carregado com sucesso!")
        model.summary()
    except Exception as e:
        logger.error(f"Erro ao tentar carregar o modelo: {e}")
        exit()

    # Prepara o gerador de dados de teste
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    # Pega os rótulos esperados e os nomes das classes
    y_true = test_generator.classes
    class_names = list(test_generator.class_indices.keys())
    logger.info(f"\nClasses para avaliação: {class_names}")

    # Faz as predições usando o modelo
    logger.info("\nIniciando predições no conjunto de teste...")
    predictions_probabilities = model.predict(test_generator, verbose=1)

    # Pega o índice da maior probabilidade
    y_pred = np.argmax(predictions_probabilities, axis=1)

    # Avaliação do desempenho do modelo
    print("\n--- Relatório Final de Classificação ---")

    # Compara os rótulos
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Calcula a acurácia do modelo
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Acurácia Geral do Modelo: {accuracy:.4f} ({accuracy:.2%})")
