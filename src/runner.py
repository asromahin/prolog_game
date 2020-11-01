from flask_ngrok import run_with_ngrok
from flask import Flask
from flask import request
from flask import render_template, send_file
app = Flask(__name__)

app.config['SECRET_KEY'] = 'kj'

GLOBAL_QUERY = 'название документа'

@app.route('/',methods = ['GET','POST'])
def index():
  if request.method == 'POST':
    img = request.files.get('img')
    text = request.form.get('text')
    print(text)
    if text is not None:
      global GLOBAL_QUERY
      GLOBAL_QUERY = text
    if img is not None:
      filename = img.filename
      path = filename
      img.save(path)
      #nim = cv2.imread(path)
      #result = pipeline(nim, query=GLOBAL_QUERY)
      return render_template('index.html', uploaded_img_name=filename, result=None)
    else:
      return render_template('index.html', uploaded_img_name=None, result=None)

  elif request.method == 'GET':
    return render_template('index.html', uploaded_img_name=None, result=None)


@app.route('/images/<filename>',methods = ['GET'])
def images(filename):
  print('send here')
  return send_file(filename)


run_with_ngrok(app)   #starts ngrok when the app is run
app.run()