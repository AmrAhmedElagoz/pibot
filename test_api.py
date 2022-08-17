import requests

url = 'http://127.0.0.1:5000/chat'
r = requests.post(url,json={"_title":"this is test title"})

print(r.json())