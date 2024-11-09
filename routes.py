from flask import Flask, render_template, request, Response, redirect, url_for, flash
import cv2
import os
import ast
from func import allowed_file, read_annotations, accident_detection

app = Flask(__name__)
app.secret_key = 'your_secret_key'  
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            video_id = filename.split('.')[0]  
            anno_path = "C:\\accident_detection_project\\data\\videos\\annotations\\Crash-1500.txt"  
            annotations = read_annotations(anno_path)
            labels = [anno['label'] for anno in annotations if anno['vid'] == video_id][0]
            
            return Response(accident_detection(filepath, labels), mimetype='multipart/x-mixed-replace; boundary=frame')

    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    video_id = "000001"  
    video_path = os.path.join("C:\\accident_detection_project\\data\\videos\\Normal", f"{video_id}.mp4")
    anno_path = "C:\\accident_detection_project\\data\\videos\\annotations\\Crash-1500.txt"
    
    annotations = read_annotations(anno_path)
    labels = next((anno['label'] for anno in annotations if anno['vid'] == video_id), [])
    return Response(accident_detection(video_path, labels), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template('predict.html') 
@app.route('/result', methods=['GET', 'POST'])
def result():
    return render_template('result.html')  
if __name__ == '__main__':
    app.run(debug=True)
