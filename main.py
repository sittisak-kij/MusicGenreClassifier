import os
# import magic
import urllib.request
from app import app
from flask import Flask, flash, request, redirect, render_template, url_for, json
from werkzeug.utils import secure_filename

import librosa
import numpy as np
from joblib import load
import os
from sklearn.preprocessing import StandardScaler
from pydub import AudioSegment

ALLOWED_EXTENSIONS = set(['mp3'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    path = os.getcwd() + '\\uploads'
    songs = os.listdir(path)
    return render_template('upload.html', songs=songs)


@app.route('/', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('File(s) successfully uploaded')
            return redirect(url_for('.result', path=filename))


@app.route('/result')
def result():
    path = request.args['path']
    result = classify(path)
    return render_template("result.html", path=path, result=result)


def classify(path):
    fullpath = os.getcwd() + '\\uploads\\' + path
    song = "song.wav"
    sound = AudioSegment.from_mp3(fullpath)
    sound.export(song, format="wav")

    SVM_MODEL = os.getcwd() + '\\static\\models\\genre-classify-20sec.lib'

    y, sr = librosa.load(song)
    t = librosa.core.get_duration(y=y)
    t = int(t)
    print(t)
    start = 0
    dur = (t // 10)
    collection = []
    for i in range(0, 10):
        print(start, dur)
        y, sr = librosa.load(song, offset=start, duration=dur)
        start += dur
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rmse(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        features = [np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff),
                    np.mean(zcr)]
        for e in mfcc:
            features.append(np.mean(e))
        collection.append(features)

    scaler = StandardScaler()
    collection = scaler.fit_transform(np.array(collection, dtype=float))

    classifier = load(SVM_MODEL)
    result = classifier.predict(collection)

    # print the genre percentage
    edm, hiphop, jazz, pop, rnb, rock = 0, 0, 0, 0, 0, 0
    for i in result:
        if i == 0:
            edm += 1
        elif i == 1:
            hiphop += 1
        elif i == 2:
            jazz += 1
        elif i == 3:
            pop += 1
        elif i == 4:
            rnb += 1
        else:
            rock += 1

    print("edm:" + str(edm * 10) + "%")
    print("hiphop: " + str(hiphop * 10) + "%")
    print("jazz: " + str(jazz * 10) + "%")
    print("pop: " + str(pop * 10) + "%")
    print("rnb: " + str(rnb * 10) + "%")
    print("rock: " + str(rock * 10) + "%")

    return [str(edm * 10), str(hiphop * 10), str(jazz * 10), str(pop * 10), str(rnb * 10), str(rock * 10)]


if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(host='0.0.0.0', port="8080", debug=True)
