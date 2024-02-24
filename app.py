import json
import cv2
import requests
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import numpy as np
import time
from html import unescape
import re
from annotate_realtime import detect_objects_and_extract_text, reset
import os

api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "AIzaSyAj5is27Ui1bJ5CMSCdGEcus41LIiZ5Zy8")
latitude, longitude = None, None
destination_location = None
last_instruction = ""
instruction_to_be_said = None

app = Flask(__name__)
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('speaker_available')
def trySpeakingNavInstruction():
    # if there is some instructions to be said, speak the instruction instantly
    print("speaker available!")
    global instruction_to_be_said
    if instruction_to_be_said is not None:
        speak(instruction_to_be_said, True)
        instruction_to_be_said = None
        return


def speak(text, important=False):
    emit("speak", {'text': text, 'important': important})


@socketio.on('send_text')
def receive_text(text):
    # Process the audio data in Python
    # Example: Convert audio data to base64 for simplicity

    # condition depending on text
    text = text.lower()
    if text == "scan" or text == "sc" or text == "activate scan" or text == "activate scanning" or text == "activate":
        speak("scan activated", True)
        emit("activate_scan")  # start the scan loop: request -> client response -> process -> request
    elif text == "quit scan" or text == "quit sc" or text == "deactivate scan" or text == "quit scanning" or text == "deactivate":
        speak("scan deactivated", True)
        emit("deactivate_scan")
    elif text[0:11] == "navigate to":
        address = text[11:]
        global destination_location
        destination_location = address
        speak(f"navigating to {destination_location}", True)
        emit("activate_navigation")
    elif text == "quit navigation" or text == "deactivate navigation" or text == "exit navigation":
        speak("navigation deactivated", True)
        global last_instruction
        last_instruction = ""
        emit("deactivate_navigation")


# image processing
@socketio.on("video_frame")
def process_image(frame):
    decoded_frame = base64.b64decode(frame)
    nparr = np.frombuffer(decoded_frame, np.uint8)

    # Decode the image using OpenCV
    try:
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        textPrompt = detect_objects_and_extract_text(img)
        if textPrompt is not None:
            speak(textPrompt)
    finally:
        emit("request_scan")


@socketio.on("scan_activated")
def startRequesting():
    print("scan_activated")
    reset()
    emit("request_scan")


@socketio.on('update_location')
def update_loc(jsonData):
    global latitude, longitude
    data = json.loads(jsonData)
    latitude = data['latitude']
    longitude = data['longitude']
    get_prompt(latitude, longitude, destination_location)
    return "Location updated successfully!"


def get_prompt(start_lat, start_long, dest_location):
    I, D, S = get_walking_directions(api_key, start_lat, start_long, dest_location)
    # closest_instruction, coords = find_closest_instruction(latitude, longitude, S, I)
    closest_instruction = I[0]
    closest_dist = D[0]
    global last_instruction
    if closest_instruction == last_instruction:
        return
    last_instruction = closest_instruction

    # distance_to_instruction = calculate_distance(latitude, longitude, coords[0], coords[1])
    # prompt = f"Next instruction: {closest_instruction}. Distance: {distance_to_instruction} km"
    instruction_text = f"{html_to_plaintext(closest_instruction)}, {closest_dist}."
    global instruction_to_be_said

    instruction_to_be_said = instruction_text


def get_walking_directions(api_key, origin_lat, origin_lng, dest_location):
    origin = f"{origin_lat},{origin_lng}"
    # destination = f"{destination_lat},{destination_lng}"
    instructions = []
    start_locations = []
    distances = []

    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={dest_location}&mode=walking&key={api_key}"
    response = requests.get(url)
    data = response.json()

    if data["status"] == "OK":

        steps = data["routes"][0]["legs"][0]["steps"]
        for step in steps:
            instruction = step["html_instructions"]
            distance = step["distance"]["text"]
            start_location = step["start_location"]
            end_location = step["end_location"]
            instructions.append(instruction)
            distances.append(distance)
            start_locations.append(start_location)
        return instructions, distances, start_locations

    else:
        speak(f"Error: {data['status']}. navigation deactivated", True)
        global last_instruction
        last_instruction = ""
        emit("deactivate_navigation")


# geocode the location
def geocode(address):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
    response = requests.get(url)
    data = response.json()

    if data["status"] == "OK":
        result = data["results"][0]
        location = result["geometry"]["location"]
        formatted_address = result["formatted_address"]
        lat = location["lat"]
        long = location["lng"]
        return lat, long, formatted_address
    else:
        return None, None, None


def html_to_plaintext(html_text):
    # Replace HTML entities with their corresponding characters
    plaintext = unescape(html_text)

    # Remove HTML tags
    plaintext = re.sub(r'<[^>]*>', ' ', plaintext)
    return plaintext


if __name__ == '__main__':
    socketio.run(app, debug=True, port=int(os.environ.get('PORT', 5000)), host='0.0.0.0')

# 51.4818048 -0.1769472
