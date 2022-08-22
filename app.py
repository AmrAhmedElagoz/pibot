from flask import Flask, request, jsonify, render_template
from model import qa
import tag_classification


app = Flask(__name__, static_url_path='/static')

@app.route('/')
def index():
    return render_template('index.html')

cache= dict()

@app.route('/chat',methods=['POST'])
def chat():
    # print(request.get_json(force=True))
    title = request.get_json(force=True)['_title']
    print(title)
    context = request.get_json(force=True)['context']
    print(context)
    cache[str(title)] = str(context)
    print(cache)

    q= 'ما هو سعر البظيخ؟'
    
    prediction = qa(q, context)   
    # prediction= 'pass'
    print(prediction)
    resp = jsonify({"Res": prediction['answer']})
    # b= resp.json
    # print(b)
    # print(b['Res'])
    resp.headers.add('Access-Control-Allow-Origin', '*')
    return resp


if __name__ == "__main__":
    app.run(debug=True)