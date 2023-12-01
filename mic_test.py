import speech_recognition
import pyttsx3
import pyaudio

recognizer = speech_recognition.Recognizer()


try:
    with speech_recognition.Microphone() as mic:
        print("Computer:","speak")
        recognizer.adjust_for_ambient_noise(mic, duration=0.5)
        audio = recognizer.listen(mic, timeout = 1)

        text = recognizer.recognize_google(audio)
        input = text.lower()
        print("USER:",input)
except speech_recognition.UnknownValueError():
    recognizer = speech_recognition.Recognizer()