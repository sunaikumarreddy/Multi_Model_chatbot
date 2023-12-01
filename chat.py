from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

import speech_recognition
import pyttsx3
import sys

#imports for hand sign detection
import pickle
import cv2
import mediapipe as mp
import numpy as np
from inference_classifier import get_video_input


#Loading Pre-trained model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Anu"
print("Let's chat! (type 'quit' to exit)")


app = Flask(__name__)#Initilisation of flask 

@app.route("/")#defines a route ("/") that renders the home page using an HTML 
def index():
    return render_template('chat.html')

#Getting input from the user
@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):

    # sentence = "do you use credit cards?"
    sentence = str(text)
    if sentence == "quit":
        sys.exit()

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                #print(f"{bot_name}: {random.choice(intent['responses'])}")
                return f"{bot_name}: {random.choice(intent['responses'])}"
    else:
        #print(f"{bot_name}: I do not understand...")
        return f"{bot_name}: I do not understand..."
    

@app.route("/mic_get", methods=["GET", "POST"])
def chat_mic():
    recognizer = speech_recognition.Recognizer()
    print("speak in mic")
    #while True:
    try:
        with speech_recognition.Microphone() as mic:
            print("Computer:","speak")
            recognizer.adjust_for_ambient_noise(mic, duration=0.5)
            audio = recognizer.listen(mic, timeout = 1)

            text = recognizer.recognize_google(audio)
            input = text.lower()
            print("USER:",input)
            response = {
                "value1": text,
                "value2": get_Chat_response_mic(input)
            }
            return jsonify(response)
    except:
        recognizer = speech_recognition.Recognizer()
        #continue
    print("end mic")


def get_Chat_response_mic(text):

    # sentence = "do you use credit cards?"
    sentence = str(text)
    if sentence == "quit":
        sys.exit()

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                #print(f"{bot_name}: {random.choice(intent['responses'])}")
                return f"{bot_name}: {random.choice(intent['responses'])}"
    else:
        #print(f"{bot_name}: I do not understand...")
        return f"{bot_name}: I do not understand..."

@app.route("/video_get", methods=["GET", "POST"])
def chat_video():
    print("I am at video get")
    video_output = get_video_input();
    print(video_output+"------")
    response = {
        "value1": video_output,
        "value2": get_Chat_response(video_output)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run()