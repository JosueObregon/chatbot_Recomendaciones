import random
import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords  # Añadimos stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Descargar los recursos de NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')  # Descargar stopwords

# Inicializamos el lematizador y las stopwords en español
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('spanish'))  # Se puede usar 'english' si es en inglés

# Cargar los datos de los intents (preguntas y respuestas)
with open('intents.json') as file:
    intents = json.loads(file.read())

# Listas para almacenar palabras, categorías y patrones (frases) clasificadas
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

# Clasificación de patrones y categorías (preprocesamiento de datos)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenización: Dividimos las oraciones en palabras
        word_list = nltk.word_tokenize(pattern)
        
        # Filtramos palabras vacías y lematizamos
        word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list if word not in stop_words]
        
        # Añadimos las palabras al conjunto general y las categorizamos
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Eliminar duplicados y ordenar palabras y clases
words = sorted(set([word for word in words if word not in ignore_letters]))
classes = sorted(set(classes))

# Guardar las palabras y clases preprocesadas para usarlas en el futuro
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Creamos el conjunto de entrenamiento: convertimos las palabras en vectores (bolsa de palabras)
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Creamos la "bolsa de palabras" (bag of words)
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Creamos la salida esperada para cada categoría (tag)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    training.append([bag, output_row])

# Mezclamos los datos de entrenamiento para evitar sesgos
random.shuffle(training)

# Convertimos a numpy arrays para trabajar con la red neuronal
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

# Dividimos los datos en conjunto de entrenamiento y validación (90% entrenamiento, 10% validación)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=42)

# Normalización de los datos para mejorar el rendimiento de la red neuronal
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
val_x = scaler.transform(val_x)

# Guardamos el scaler para futuras predicciones
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Creamos la estructura de la red neuronal
model = Sequential()

# Capa de entrada: el tamaño está determinado por la cantidad de palabras únicas (bolsa de palabras)
model.add(Input(shape=(len(train_x[0]),), name="input_layer"))

# Primera capa oculta con regularización L2 para evitar el sobreajuste
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), name="hidden_layer1"))
model.add(Dropout(0.3, name="dropout1"))  # Dropout del 30% para evitar el sobreajuste

# Segunda capa oculta
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), name="hidden_layer2"))
model.add(Dropout(0.3, name="dropout2"))

# Capa de salida con tantas neuronas como clases de salida (usamos softmax para clasificación multiclase)
model.add(Dense(len(train_y[0]), activation='softmax', name="output_layer"))

# Creamos el optimizador Adam con una tasa de aprendizaje baja
adam = Adam(learning_rate=0.0005)

# Compilamos el modelo con pérdida categórica cruzada, adecuado para clasificación multiclase
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Configuramos EarlyStopping para detener el entrenamiento si no mejora tras 10 épocas
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenamos el modelo
history = model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=1, callbacks=[early_stopping], validation_data=(val_x, val_y))

# Guardamos el modelo entrenado
model.save('chatbot_model.keras')

# Visualización de métricas (pérdida y precisión)
plt.figure(figsize=(12, 6))

# Gráfico de la pérdida en el entrenamiento y validación
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss (Train)')
plt.plot(history.history['val_loss'], label='Loss (Val)')
plt.title('Pérdida (Loss)')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()

# Gráfico de la precisión en el entrenamiento y validación
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy (Train)')
plt.plot(history.history['val_accuracy'], label='Accuracy (Val)')
plt.title('Precisión (Accuracy)')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()

plt.show()
