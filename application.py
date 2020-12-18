from flask import Flask, request, redirect, render_template, flash, url_for, send_from_directory
import os
import pickle
from tensorflow import keras

from generate_captions import generate_captions

# from werkzeug import secure_filename
cap_model = keras.models.load_model("./model_files/Image_caption_v1.h5")
with open('./model_files/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('./model_files/features2.pkl', 'rb') as f:
    img_features = pickle.load(f)

UPLOAD_FOLDER = 'static/uploads/'

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route('/')
def display_root():
    return render_template('upload.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
        print(f.filename)

        return redirect(url_for('uploaded_file', filename=f.filename))


@ app.route('/show/<filename>')
def uploaded_file(filename):
    new_caption = generate_captions(model=cap_model, image=filename, tokenizer=tokenizer,
                                    max_caption_length=34, feature_vector=img_features)
    print(new_caption)
    return render_template('template.html', filename=filename, caption=new_caption)


@ app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
