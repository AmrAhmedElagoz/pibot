import nltk
from nltk.stem import WordNetLemmatizer

import json
import pickle
import random

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from keras.models import load_model
import tensorflow as tf
# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.set_visible_devices([], 'GPU')

lemmatizer = WordNetLemmatizer()
ignore_words = ['?', '!']

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
intents = json.loads(open('intents_ar.json', encoding="utf8").read())

model = load_model('chatbot_model_v2.h5')


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
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(predictions, intents_json):
    # check if there is any response
    # print(ints)
    if len(predictions) == 0:
      result = random.choice(intents_json['intents'][2]['responses'])
      return result

    list_of_intents = intents_json['intents']
    tag = predictions[0]['intent']

    print(predictions[0])
    
    # check if prob of response is lower the 40% 
    if float(predictions[0]['probability']) < 0.5:
      # if the tag classification is one of the essential tags -> respond by "مش فاهمك" 
      if tag == "greeting" or tag == "thanks" or tag == "noanswer" or tag == "goodbye" or tag == "Confirmations" or tag == "Rejections":
        result = random.choice(list_of_intents[2]['responses'])
        return result
      # if any other tags --> ask to talk about the topic with another way
      else:
        # return  "؟" + tag + " انت بتتكلم عن الـ"
        return  "ممكن تقولى السؤال بصيغة مختلفة ،" + tag + " انا فهمت انك بتتكلم عن الـ"
    

    result = None
    for i in list_of_intents:
        if(i['tag']== tag):
            if tag == "greeting" or tag == "thanks" or tag == "noanswer" or tag == "goodbye" or tag == "Confirmations" or tag == "Rejections":
                result = random.choice(i['responses'])
                break
            else:
                #### MODEL HERE
                break
    return result



def chatbot_response(msg, intents_):
    predictions = predict_class(msg, model)
    res = getResponse(predictions, intents_)
    return res, predictions




def send():
  print('صباح الجمال والكريستال، انا اخوك الشات بوت اؤمرنى اساعدك ازاى؟')
  print('=========================================')
  isWorking = True
  while isWorking:
    msg = str(input("You: "))
    res, ints = chatbot_response(msg, intents)
    if len(ints) == 0:
      print("bot: " + res + '\n')
      continue
    pred = ints[0]['intent']

    if msg == 'exit' or msg == 'quit' or msg == 'خروج' or msg == 'انهاء':
      isWorking = False
      break
    if pred == 'goodbye':
      print("bot: " + res + '\n')
      isWorking = False
      break
    print("bot: " + res + '\n')