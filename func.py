from flask import Flask, render_template, request, Response, redirect, url_for, flash
import cv2
import os
import ast

ALLOWED_EXTENSIONS = {'mp4', 'avi'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_annotations(anno_path):
    annotations = []
    with open(anno_path, 'r') as file:
        for line in file:
            entry = line.strip().split()  # Assuming space-separated entries
            if len(entry) > 1:
                try:
                    labels = ast.literal_eval(entry[1])  # Convert the string representation of a list to a list
                    video_id = entry[0]  # Assuming the first element is the video ID
                    annotations.append({'vid': video_id, 'label': [int(x) for x in labels]})  # Create a dict
                except (ValueError, SyntaxError) as e:
                    print(f"Error processing line '{line}': {e}")
    return annotations


def accident_detection(video_file, labels, topN=50):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= topN:
            break
        label = labels[frame_count] if frame_count < len(labels) else 0
        if label == 1:
            cv2.putText(frame, 'Accident', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'Normal', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Encode the frame for Flask streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        frame_count += 1
    cap.release()