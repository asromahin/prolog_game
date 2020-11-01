from flask import Flask, render_template, request, redirect, url_for
from get_images import get_images, get_path, get_directory
from get_prediction import get_prediction
from generate_html import generate_html
from torchvision import models
import json

app = Flask(__name__)


# Make sure to pass `pretrained` as `True` to use the pretrained weights:
path_model = ''
model = torch.load(path_model)
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()


def get_image_class(path):
    get_images(path)
    path = get_path(path)
    tag_image, tag_titul = get_prediction(model, path)
    tag = tag_image+':'+tag_titul
    print(tag)
    generate_html(tag)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        user = request.form['search']
        get_image_class(user)
        return redirect(url_for('success', name=get_directory(user)))


@app.route('/success/<name>')
def success(name):
    return render_template('image_class.html')


if __name__ == '__main__' :
    app.run(debug=True)
