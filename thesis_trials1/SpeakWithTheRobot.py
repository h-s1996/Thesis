#coding: utf-8
import speech_recognition as sr
from gtts import gTTS
from sys import stdin
from os import system


class SpeakWithTheRobot:
    def __init__(self, human_lsa, naives, human_keywords, robot_vectors):
        self.slow = False
        self.device_id = 0
        self.lang = 'pt-pt'
        self.naives = naives
        self.chunk_size = 2048
        self.r = sr.Recognizer()
        self.sample_rate = 48000
        self.human_lsa = human_lsa
        self.robot_vectors = robot_vectors
        self.human_keywords = human_keywords

    def hear(self):
        with sr.Microphone(device_index=self.device_id, sample_rate=self.sample_rate, chunk_size=self.chunk_size) as source:
            self.r.adjust_for_ambient_noise(source)
            print("Say Something")
            audio = self.r.listen(source)
            try:
                text = self.r.recognize_google(audio, language="pt-PT")
                print("you said: " + text)
                return text
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service;{0}".format(e))

    def speak(self, phrase):
        tts = gTTS(text=phrase, lang=self.lang, slow=self.slow)
        tts.save("soundfile.mp3")
        system("mpg123 soundfile.mp3")
        return

    def speaking_to_the_robot(self):
        while True:
            print("Press a character")
            c = stdin.read(2)
            if c[0] == 's':
                self.speak(self.robot_vectors.search_for_phrase(self.human_lsa, self.naives, self.hear(),
                                                                self.human_keywords))
            elif c[0] == 't':
                print(self.robot_vectors.search_for_phrase(self.human_lsa, self.naives, "Olá", self.human_keywords))
                print(self.robot_vectors.search_for_phrase(self.human_lsa, self.naives, "Como está tudo a andar?",
                                                           self.human_keywords))
                print(self.robot_vectors.search_for_phrase(self.human_lsa, self.naives, "Comigo está tudo fantástico.",
                                                           self.human_keywords))
                print(self.robot_vectors.search_for_phrase(self.human_lsa, self.naives, "Gosto muito de vaca",
                                                           self.human_keywords))
                print(self.robot_vectors.search_for_phrase(self.human_lsa, self.naives, "Gosto de estar com a minha filha.",
                                                           self.human_keywords))
                print(self.robot_vectors.search_for_phrase(self.human_lsa, self.naives, "Uma das minhas coisas preferidas é passear em parques.",
                                                           self.human_keywords))
            elif c[0] == 'q':
                break
