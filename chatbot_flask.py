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
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, jsonify

# Descargar los recursos de NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Inicializamos el lematizador y las stopwords en español
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('spanish'))

# Cargar los datos de los intents (preguntas y respuestas)
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

# Listas para almacenar palabras, categorías y patrones (frases) clasificadas
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

# Clasificación de patrones y categorías (preprocesamiento de datos)
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        word_list = [lemmatizer.lemmatize(word.lower()) for word in word_list if word not in stop_words]
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = sorted(set([word for word in words if word not in ignore_letters]))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=42)

scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
val_x = scaler.transform(val_x)

pickle.dump(scaler, open('scaler.pkl', 'wb'))

model = Sequential()
model.add(Input(shape=(len(train_x[0]),), name="input_layer"))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), name="hidden_layer1"))
model.add(Dropout(0.3, name="dropout1"))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), name="hidden_layer2"))
model.add(Dropout(0.3, name="dropout2"))
model.add(Dense(len(train_y[0]), activation='softmax', name="output_layer"))

adam = Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=1, callbacks=[early_stopping], validation_data=(val_x, val_y))

model.save('chatbot_model.keras')

# Funciones para procesar y predecir respuestas
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow_input = bow(sentence, words)
    res = model.predict(np.array([bow_input]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Inicialización de la aplicación Flask
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# Variables globales para llevar el seguimiento de preguntas y respuestas
current_question = 0
user_responses = {}

@app.route("/get")
def chatbot_response():
    global current_question, user_responses
    msg = request.args.get('msg')
    
    # Si es la primera pregunta o una respuesta del usuario
    if current_question == 0 or msg.lower() not in ['hola', 'empezar']:
        user_responses[intents['intents'][current_question]['tag']] = msg
        current_question += 1

    # Si hay más preguntas, envía la siguiente
    if current_question < len(intents['intents']):
        response = intents['intents'][current_question]['responses'][0]
    else:
        # Procesa las respuestas del usuario y da una recomendación final
        response = "Gracias por responder. Basado en tus respuestas, te recomendaría..."
        print(user_responses)  # Para debug, imprime las respuestas en la consola
        
        # Reiniciar para la próxima conversación
        current_question = 0
        user_responses = {}

    return response


if __name__ == "__main__":
    app.run(debug=True)