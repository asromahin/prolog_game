from src.pipeline import Pipeline
from flask_ngrok import run_with_ngrok
from flask import Flask
from flask import request
from flask import render_template, send_file
from PIL import Image
import os

#GLOBAL_QUERY = None

def init_app(pipeline,  template_folder='prolog_game', global_text='тип документа'):
  app = Flask(__name__, template_folder=template_folder)

  app.config['SECRET_KEY'] = 'kj'

  @app.route('/', methods=['GET', 'POST'])
  def index():
    if request.method == 'POST':
      img = request.files.get('img')
      text = request.form.get('text')
      print(text)
      if text is not None:
        global_text = text
      if img is not None:
        filename = img.filename
        print(filename)
        path = filename
        img.save(path)
        nim = Image.open(path)
        nim = nim.convert('RGB')
        result = pipeline.predict(nim, query=global_text)
        return render_template('index.html', uploaded_img_name=filename, result=result['ner_result'][0][0])
      else:
        return render_template('index.html', uploaded_img_name=None, result=None)

    elif request.method == 'GET':
      return render_template('index.html', uploaded_img_name=None, result=None)

  @app.route('/images/<filename>', methods=['GET'])
  def images(filename):
    print('send here')
    print(filename)
    return send_file(filename)
  run_with_ngrok(app)   #starts ngrok when the app is run
  app.run()