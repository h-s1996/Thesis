#coding: utf-8
import speech_recognition as sr
from gtts import gTTS
from sys import stdin
from os import system


class SpeakWithTheRobot:
    """
    A class created to enable the user to have a conversation with the "robot" utilizing the system proposed.
    ...

    Methods
    -------
    listen()
        The software listens to a human and transcribes what he/she said into text.
    speak(utterance)
        The software utters an utterance through the Google Text to Speech package using a sound file.
    speaking_to_the_robot(lsa, naive, file)
        It enables an user to have a conversation with the "robot" using the system proposed. It is also permitted to
        check the result of a couple of testing phrases.
    """

    def __init__(self):
        self.slow = False
        self.device_id = 0
        self.lang = 'pt-pt'
        self.chunk_size = 2048
        self.r = sr.Recognizer()
        self.sample_rate = 48000

    def listen(self):
        """
        The software listens to a human and transcribes what he/she said into text.
        :return: a text that represents the audio utter by the human.
        :rtype: str
        """
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

    def speak(self, utterance):
        """
        The software utters an utterance through the Google Text to Speech package using a sound file.
        :param utterance:
        :type utterance: str
        """
        tts = gTTS(text=utterance, lang=self.lang, slow=self.slow)
        tts.save("soundfile.mp3")
        system("soundfile.mp3")

    def speaking_to_the_robot(self, lsa, naive, db):
        """
        It enables an user to have a conversation with the "robot" using the system proposed. It is also permitted to
        check the result of a couple of testing phrases.
        :param lsa: a Latent Semantic Analysis object
        :type lsa: LSA
        :param naive: a Naive Bayes classifier
        :type naive: NaiveBayesClassifier
        :param db: an object that represents the database and it is connected to the Database file
        :type db: Database
        """
        while True:
            print("Press a character")
            c = stdin.read(2)
            if c[0] == 's':
                self.speak(db.get_robot_utterance(naive.predict_new_robot_id(
                    lsa.process_new_human_utterance(self.listen(), db.human_utterances))))
            elif c[0] == 't':
                print(db.get_robot_utterance(naive.predict_new_robot_id(
                    lsa.process_new_human_utterance("Bom dia", db.human_utterances))))
                print(db.get_robot_utterance(naive.predict_new_robot_id(
                    lsa.process_new_human_utterance("Como está tudo a andar?", db.human_utterances))))
                print(db.get_robot_utterance(naive.predict_new_robot_id(
                    lsa.process_new_human_utterance("Comigo está tudo fantástico.", db.human_utterances))))
                print(db.get_robot_utterance(naive.predict_new_robot_id(
                    lsa.process_new_human_utterance("Gosto muito de vaca", db.human_utterances))))
                print(db.get_robot_utterance(naive.predict_new_robot_id(
                    lsa.process_new_human_utterance("Gosto de estar com a minha filha.", db.human_utterances))))
                print(db.get_robot_utterance(naive.predict_new_robot_id(
                    lsa.process_new_human_utterance("Uma das minhas coisas preferidas é passear em parques.",
                                                    db.human_utterances))))
            elif c[0] == 'q':
                break
