import os
import cv2
import pygame
import random
import speech_recognition as sr
import pyttsx3
import time
import subprocess
import pyttsx3
import time
import pathlib
import textwrap
import speech_recognition as sr
import google.generativeai as genai

from IPython.display import Markdown

# Load pre-trained Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Define folder paths for different emotions
folder_paths = {
    "Depression": r"C:\Users\divit\OneDrive\Desktop\epic project\epics\song\depression",
    "Anxiety": r"C:\Users\divit\OneDrive\Desktop\epic project\epics\song\anxiety",
    "Sad": r"C:\Users\divit\OneDrive\Desktop\epic project\epics\song\sad",
}
video_path = r"C:\Users\divit\OneDrive\Desktop\epic project\epics\song\video"

# Dummy dataset containing questions
questions = [
    "How are you feeling today?",
    "What's on your mind?",
    "Tell me about your day.",
    "Do you have any worries?",
    "What makes you happy?"
]

def detect_emotion_from_image(gray):
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    # Initialize variables to track emotions
    sadness_count = 0
    anxiety_count = 0
    depression_count = 0

    for (x, y, w, h) in faces:
        # Extract region of interest (ROI) for face
        roi_gray = gray[y:y+h, x:x+w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Check for the presence of closed eyes
        if len(eyes) == 0:
            anxiety_count += 1

        # Check for the presence of multiple faces
        if len(faces) == 1:
            sadness_count += 1

    # Determine the dominant emotion
    if sadness_count > 0:
        dominant_emotion = "Sad"
    elif anxiety_count > 0:
        dominant_emotion = "Anxiety"
    else:
        dominant_emotion = "Depression"  # Default label

    # Print the dominant emotion to the terminal
    print(f"Dominant emotion from image: {dominant_emotion}")

    return dominant_emotion

def detect_emotion_from_audio():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Speak into the microphone to detect emotion...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source, timeout=10)

    try:
        print("Processing audio...")
        text = recognizer.recognize_google(audio)
        print(f"Recognized text: {text}")

        # Implement emotion identification logic here
        # For simplicity, this example just returns a random choice
        return random.choice(["Depression", "Anxiety", "Sad"])

    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Error: {e}")
        return None

def play_songs_with_video(emotion, folder_paths, video_path):
    if emotion in folder_paths:
        folder_path = folder_paths[emotion]
        if os.path.exists(folder_path):
            songs = [os.path.join(folder_path, song) for song in os.listdir(folder_path)]
            if songs:
                print(f"Playing songs randomly from {emotion} folder with video. Press 'q' to stop.")
                pygame.init()
                pygame.mixer.init()
                cap = cv2.VideoCapture(os.path.join(video_path, random.choice(os.listdir(video_path))))

                # Get the original frame rate of the video
                frame_rate = cap.get(cv2.CAP_PROP_FPS)

                try:
                    random.shuffle(songs)
                    for song in songs:
                        print(f"Playing: {song}")
                        pygame.mixer.music.load(song)
                        pygame.mixer.music.play()

                        while pygame.mixer.music.get_busy() or cap.isOpened():
                            ret, frame = cap.read()
                            if ret:
                                cv2.imshow('Video Playback', frame)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                                # Introduce delay to match the original frame rate
                                time.sleep(1 / frame_rate)
                            else:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                    cap.release()
                    pygame.quit()
                    cv2.destroyAllWindows()
                except KeyboardInterrupt:
                    cap.release()
                    pygame.quit()
                    cv2.destroyAllWindows()
                    print("Playback stopped by user.")
            else:
                print(f"No songs found in {emotion} folder.")
        else:
            print(f"{emotion} folder not found.")
    else:
        print(f"Emotion '{emotion}' not recognized.")

def run_another_python_file(file_path):
    try:
        subprocess.run(["python", file_path])
    except Exception as e:
        print(f"Error running Python file: {e}")

def text_to_speech(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    #for question in questions:
       # print(f"Question: {question}")
        # Emotion detection from image (video feed)
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        detected_emotion_from_image = None
        while time.time() - start_time < 10:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_emotion_from_image = detect_emotion_from_image(gray)
            cv2.putText(frame, detected_emotion_from_image, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Emotion Detection from Image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        # Emotion detection from audio
        detected_emotion_from_audio = detect_emotion_from_audio()
        if detected_emotion_from_audio:
            print(f"Detected emotion from audio: {detected_emotion_from_audio}")
            text_to_speech(f"The detected emotion from audio is {detected_emotion_from_audio}")

            # Prompt user for input
            print("Select an option:")
            print("1. Play songs and video")
            print("2. Run another Python file")
            option = input("Enter option: ")

            if option == '1':
                play_songs_with_video(detected_emotion_from_audio, folder_paths, video_path)
            elif option == '2':
                    def recognize_speech():
                        recognizer = sr.Recognizer()
                        microphone = sr.Microphone()

                        with microphone as source:
                            recognizer.adjust_for_ambient_noise(source)
                            print("Listening for speech...")
                            audio = recognizer.listen(source)

                        try:
                            print("Recognizing speech...")
                            text = recognizer.recognize_google(audio)
                            print(f"You said: {text}")
                            return text
                        except sr.RequestError:
                            print("API was unreachable or unresponsive")
                        except sr.UnknownValueError:
                            return "Unable to recognize speech"

                    def text_to_speech(text):
                        engine = pyttsx3.init()
                        engine.say(text)
                        engine.runAndWait()

                    def to_markdown(text):
                        text = text.replace('•', '  *')
                        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

                    GOOGLE_API_KEY="AIzaSyArPn-IK2Qeu2iilkBE-ZNyaymWqegYOAY"

                    genai.configure(api_key=GOOGLE_API_KEY)

                    model = genai.GenerativeModel('gemini-pro')
                    chat = model.start_chat(history=[])
                    response = chat.send_message("Hi Gemini, I have some questions. Please reply with short answers, one or two lines each. if the given prompt is like a stop or end the conversation return True")

                    while True:
                        print()
                        print("Ask the question  :")
                        ques = recognize_speech()
                        response = chat.send_message(ques)
                        if response.text.lower()=='true':
                            print('Thank you! Have a good day ...')
                            text_to_speech('Thank you! Have a good day ...')
                            break
                        a=to_markdown(response.text)
                        print()
                        print("answer :",response.text,sep=" ")
                        time.sleep(1)
                        text_to_speech(response.text)   
                        time.sleep(1)
            else:
                print("Invalid option. Please select 1 or 2.")
        else:
            print("No emotion detected from audio.")