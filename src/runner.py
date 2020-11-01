from src.pipeline import Pipeline
from flask_ngrok import run_with_ngrok
from flask import Flask
from flask import request
from flask import render_template, send_file
from PIL import Image

def init_app(cls_model_path, credential_path):
  app = Flask(__name__)

  app.config['SECRET_KEY'] = 'kj'

  pipeline = Pipeline(cls_model_path=cls_model_path, credential_path=credential_path)

  GLOBAL_QUERY = 'название документа'
  @app.route('/',methods = ['GET', 'POST'])
  def index():
    global GLOBAL_QUERY
    if request.method == 'POST':
      img = request.files.get('img')
      text = request.form.get('text')
      print(text)
      if text is not None:

        GLOBAL_QUERY = text
      if img is not None:
        filename = img.filename
        path = filename
        img.save(path)
        nim = Image.open(path)
        result = pipeline.predict(nim, query=GLOBAL_QUERY)
        return render_template('index.html', uploaded_img_name=filename, result=result['ner_result'][0][0], global_query=GLOBAL_QUERY)
      else:
        return render_template('index.html', uploaded_img_name=None, result=None, global_query=GLOBAL_QUERY)

    elif request.method == 'GET':
      return render_template('index.html', uploaded_img_name=None, result=None, global_query=None)


  @app.route('/images/<filename>',methods = ['GET'])
  def images(filename):
    print('send here')
    return send_file(filename)
  run_with_ngrok(app)   #starts ngrok when the app is run
  app.run()