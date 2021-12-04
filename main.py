import discord
import json
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import  MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from discord.ext import commands
import nltk
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from datetime import datetime
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intent.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
list = [
            "You are <HUMAN>! How can I help?",
            "Your name is  <HUMAN>, how can I help you?",
            "They call you <HUMAN>, what can I do for you?",
            "Your name is <HUMAN>, how can I help you?",
            "<HUMAN>, what can I do for you?",
            "Great! Hi <HUMAN>! How can I help?",
            "Good! Hi <HUMAN>, how can I help you?",
            "Cool! Hello <HUMAN>, what can I do for you?",
            "OK! Hola <HUMAN>, how can I help you?",
            "OK! hi <HUMAN>, what can I do for you?"
            ]
anotherlist = ["One moment",
            "One sec",
            "One second",
            ]
list3 = ["Let me see",
        "Please look at the camera"]
bot = commands.Bot(command_prefix=".", description="e",
                   intents=discord.Intents.all())

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['intent']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res
    




@bot.event
async def on_ready():
    print('ai online')

@bot.event
async def on_message(message):
    if message.author.bot:
        return
    res = chatbot_response(message.content)
    if message.channel.id == 867734448514138153:
        if res in list:
            new = res.replace('<HUMAN>', message.author.name)
            await message.channel.send(new)
            return
        if res in list3:
            await message.channel.send(res)
            await message.channel.send(random.choice([f"Hi {message.author.name}, how are you?",
                              f"I believe you are {message.author.name}, how are you?",
                              f"You are {message.author.name}, how are you doing?"]))
            return
        if res in anotherlist:
            await message.channel.send(res)
            await message.channel.send(random.choice([f"The time is {datetime.utcnow()} in UTC",
                              f"Right now it is {datetime.utcnow()} in UTC",
                              f"It is around {datetime.utcnow()} in UTC"]))
            return
        else:
            await message.channel.send(res)



bot.run('tkn')

