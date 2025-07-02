import os
import sys
import tensorflow as tf
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

from services.embedding_service import EmbeddingService

# Muda para o diretório raiz do projeto
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)

# Caminhos
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data/train/")
TEST_DIR = os.path.join(PROJECT_ROOT, "data/test/")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/libras_embedding_model2.h5")

# Recupera os labels usados no treinamento
class_indices = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
    "I": 7, "L": 8, "M": 9, "N": 10, "O": 11, "P": 12,
    "Q": 13, "R": 14, "S": 15, "T": 16, "U": 17, "V": 18,
    "W": 19, "Y": 20
}

def get_label_safely(path, class_dict):
    try:
        # Get the class folder name from the full path
        path_parts = os.path.normpath(path).split(os.sep)
        # Find the index after 'train' or 'test' in the path
        data_index = path_parts.index('data')
        class_folder = path_parts[data_index + 2]  # +2 because it's data/train|test/CLASS
        label_char = class_folder.upper()

        if label_char in class_dict:
            return class_dict[label_char]
        else:
            print(f"Warning: Class '{label_char}' not found in class indices")
            return None
    except (IndexError, ValueError) as e:
        print(f"Warning: Could not process path {path}: {str(e)}")
        return None

class_names = list(class_indices.keys())
num_classes = len(class_names)

# 1. Inicializa o serviço de embedding com o caminho correto do modelo
embedding_service = EmbeddingService()
embedding_service.embedding_extractor.model = tf.keras.models.load_model(MODEL_PATH)

# 2. Prepara os geradores de dados
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

print("\nTraining data classes:", train_generator.class_indices)
print("Test data classes:", test_generator.class_indices)

print("Extraindo embeddings do conjunto de treino...")
# 3. Extrai embeddings do conjunto de treino
train_embeddings = []
train_labels = []

for i in range(len(train_generator)):
    batch_paths = [f"{TRAIN_DIR}{train_generator.filenames[j]}"
                   for j in range(i * train_generator.batch_size,
                                  min((i + 1) * train_generator.batch_size, len(train_generator.filenames)))]

    if not batch_paths:
        break

    batch_embeddings = embedding_service.embedding_extractor.extract_embeddings_batch(batch_paths)
    valid_samples = []
    valid_embeddings = []

    for idx, path in enumerate(batch_paths):
        label = get_label_safely(path, class_indices)
        if label is not None:
            train_labels.append(label)
            valid_embeddings.append(batch_embeddings[idx])

    train_embeddings.extend(valid_embeddings)

print("Extraindo embeddings do conjunto de teste...")
test_embeddings = []
test_labels = []

for i in range(len(test_generator)):
    batch_paths = [f"{TEST_DIR}{test_generator.filenames[j]}"
                   for j in range(i * test_generator.batch_size,
                                  min((i + 1) * test_generator.batch_size, len(test_generator.filenames)))]

    if not batch_paths:
        break

    batch_embeddings = embedding_service.embedding_extractor.extract_embeddings_batch(batch_paths)
    valid_samples = []
    valid_embeddings = []

    for idx, path in enumerate(batch_paths):
        label = get_label_safely(path, class_indices)
        if label is not None:
            test_labels.append(label)
            valid_embeddings.append(batch_embeddings[idx])

    test_embeddings.extend(valid_embeddings)

# 5. Converte para arrays numpy
X_train = np.array(train_embeddings)
y_train = np.array(train_labels)
X_test = np.array(test_embeddings)
y_test = np.array(test_labels)

# Print shape and unique values for debugging
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Unique training labels: {np.unique(y_train)}")
print(f"Unique test labels: {np.unique(y_test)}")

# 6. Treina o KNN
print("\nTreinando o classificador KNN...")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 7. Faz as predições
y_pred = knn.predict(X_test)

# 8. Calcula e exibe as métricas
print("\n Classification Report:")
print(classification_report(y_test, y_pred, labels=range(num_classes), target_names=class_names))

# 9. Calcula precision e recall para cada classe
precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, labels=range(num_classes))

print("\n Métricas por classe:")
for i, classe in enumerate(class_names):
    print(f"\nClasse {classe}:")
    print(f"Precision: {precision[i]:.4f}")
    print(f"Recall: {recall[i]:.4f}")
    print(f"F1-score: {f1_score[i]:.4f}")
    print(f"Support: {support[i]}")

# 10. Calcula as métricas médias
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
print(f"Macro-averaged Precision: {precision_macro:.4f}")
print(f"Macro-averaged Recall: {recall_macro:.4f}")
print(f"Macro-averaged F1-score: {f1_macro:.4f}")

# 11. Calcula a acurácia geral
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Acurácia geral: {accuracy:.4f}")
