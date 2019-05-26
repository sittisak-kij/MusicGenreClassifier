from flask import Flask
import os

UPLOAD_FOLDER = os.getcwd() + '\\uploads'

app = Flask(__name__)
app.run(host= '0.0.0.0', port="80")
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024