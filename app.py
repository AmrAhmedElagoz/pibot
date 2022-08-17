import numpy as np
from flask import Flask, request, jsonify, render_template
from transformers import pipeline

app = Flask(__name__, static_url_path='/static')
# model = pipeline('question-answering',model='ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat',methods=['POST'])
def chat():
    # print(request.get_json(force=True))
    title = request.get_json(force=True)['_title']
    print(title)
    context = request.get_json(force=True)['context']
    print(context)

    # prediction = model.predict([context])   
    prediction= 'pass'
    print(prediction)
    resp = jsonify({"Res": prediction})
    # b= resp.json
    # print(b)
    # print(b['Res'])
    resp.headers.add('Access-Control-Allow-Origin', '*')
    return resp


if __name__ == "__main__":
    app.run(debug=True)