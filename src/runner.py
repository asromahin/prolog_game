from src.pipeline import Pipeline
from flask_ngrok import run_with_ngrok
from flask import Flask
from flask import request
from flask import render_template, send_file
from PIL import Image
import os

#GLOBAL_QUERY = None

def init_app(cls_model_path, credential_path, template_folder='prolog_game'):
  #global GLOBAL_QUERY
  #GLOBAL_QUERY = 'название документа'
  app = Flask(__name__, template_folder=template_folder)

  app.config['SECRET_KEY'] = 'kj'

  pipeline = Pipeline(cls_model_path=cls_model_path, credential_path=credential_path)

  @app.route('/',methods = ['GET', 'POST'])
  def index():
    #global GLOBAL_QUERY
    if request.method == 'POST':
      img = request.files.get('img')
      text = request.form.get('text')
      print(text)
      #if text is not None:
        #global GLOBAL_QUERY
        #GLOBAL_QUERY = text
      if img is not None:
        #global GLOBAL_QUERY
        filename = img.filename
        path = filename
        img.save(path)
        nim = Image.open(path)
        global pipeline
        result = pipeline.predict(nim, query='тип документа')
        return render_template('index.html', uploaded_img_name=filename, result=result['ner_result'][0][0], global_query=None)
      else:
        return render_template('index.html', uploaded_img_name=None, result=None, global_query=None)

    elif request.method == 'GET':
      return render_template('index.html', uploaded_img_name=None, result=None, global_query=None)


  @app.route('/images/<filename>',methods = ['GET'])
  def images(filename):
    print('send here')
    return send_file(filename)
  run_with_ngrok(app)   #starts ngrok when the app is run
  app.run()