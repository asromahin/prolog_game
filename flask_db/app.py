from flask import Flask, render_template, request, json
import json
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/filter', methods=['POST'])
def filter():
    data = pd.read_excel('C:/Users/adsieg/Desktop/flask_tuto/wine_filter.xlsx', encoding='utf8')
    data = data[(data['Документ']==request.form["Кем выдан"]) & (data['Кем выдан']==request.form["Документ"])]
    data = data.head(50)
    data = data.to_dict(orient='records')
    response = json.dumps(data, indent=2)
    return response

if __name__ == '__main__':
    app.run(host='localhost', port=8089)