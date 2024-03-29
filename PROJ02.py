from __future__ import division

import re
import sys
import os

from google.cloud import speech
from google.cloud.speech import enums as enums_speech
from google.cloud.speech import types as types_speech
from google.cloud import language
from google.cloud.language import enums as enums_lang
from google.cloud.language import types as types_lang
import pyaudio #for recording audio!
import pygame  #for playing audio
from six.moves import queue

from gtts import gTTS
import os
import time
from adafruit_crickit import crickit
from adafruit_seesaw.neopixel import NeoPixel
 
num_pixels = 18  # Number of pixels driven from Crickit NeoPixel terminal
 
# The following line sets up a NeoPixel strip on Seesaw pin 20 for Feather
pixels = NeoPixel(crickit.seesaw, 20, num_pixels)

# Audio recording parameters, set for our USB mic.
RATE = 48000 # this should be 48000 for new mic; 44100 for old
CHUNK = int(RATE / 10)  # 100ms
greeted = False

credential_path = "/home/pi/DET-2019-d3b82e6383ae.json" #replace with your file name!
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=credential_path

client_s2t = speech.SpeechClient()
client_lang = language.LanguageServiceClient()

pygame.init()
pygame.mixer.init()

# servo nicknames
REACTION_SERVO = crickit.servo_1
CUMULATIVE_SERVO = crickit.servo_2
POT_SERVO = crickit.servo_3

# original servo angles
original_reaction_angle = 110
original_cumulative_angle = 90
original_pot_angle = 25
cumulative_delta_angle = 30
cumulative_max_angle = 130

# colors
RED = (255, 0, 0)
YELLOW = (255, 150, 0)
GREEN = (0, 255, 0)
CYAN = (0, 255, 255)
BLUE = (0, 0, 255)
PURPLE = (180, 0, 255)
OFF = (0,0,0)

#MicrophoneStream() is brought in from Google Cloud Platform
class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)

#this loop is where the microphone stream gets sent
def listen_print_loop(responses):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript
        
        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
#            sys.stdout.write(transcript + overwrite_chars + '\r')
#            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)
            #transcript = "oh no I'm running late"
            # Make a document from the transcript (needed for sentiment analysis?)
            type_ = enums_lang.Document.Type.PLAIN_TEXT
            language = "en"
            document = {"content": transcript, "type": type_, "language": language}
            
            # Make a 'response' to get a sentiment score
            encoding_type = enums_lang.EncodingType.UTF8
            response = client_lang.analyze_sentiment(document, encoding_type=encoding_type)
            #response = "To the person who pleases Him, God gives wisdom, knowledge and happiness, but to the sinner he gives the task of gathering and storing up wealth to hand it over to the one who pleases God. This too is meaningless, a chasing after the wind. I am not saying this because I am in need, for I have learned to be content whatever the circumstances. I know what it is to be in need, and I know what it is to have plenty. I have learned the secret of being content in any and every situation, whether well fed or hungry, whether living in plenty or in want. I can do all things through Christ who gives me strength."
            sentiment = response.document_sentiment.score
            magnitude = response.document_sentiment.magnitude
            print("Document sentiment score: {}".format(sentiment))
            print("Document magnitude score: {}".format(magnitude))
            #if there's a voice activitated quit - quit!
            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                print('Exiting..')
                break
            else:
                decide_action(transcript, sentiment)
#            print(transcript)
            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            num_chars_printed = 0

def decide_action(transcript, sentiment):
    '''
    transcript = "To the person who pleases Him, God gives wisdom, knowledge and happiness, but to the sinner he gives the task of gathering and storing up wealth to hand it over to the one who pleases God. This too is meaningless, a chasing after the wind. I am not saying this because I am in need, for I have learned to be content whatever the circumstances. I know what it is to be in need, and I know what it is to have plenty. I have learned the secret of being content in any and every situation, whether well fed or hungry, whether living in plenty or in want. I can do all things through Christ who gives me strength."
    document = {"content": transcript, "type": enums_lang.Document.Type.PLAIN_TEXT, "language": "en"}
    encoding_type = enums_lang.EncodingType.UTF8
    response = client_lang.analyze_sentiment(document, encoding_type=encoding_type)
    sentiment = response.document_sentiment.score
    magnitude = response.document_sentiment.magnitude
    print("example sentiment score: {}".format(sentiment))
    print("example magnitude score: {}".format(magnitude))
    '''
    #here we're using some simple code on the final transcript from
    #GCP to figure out what to do, how to respond.
    global greeted
    
    if re.search('good morning',transcript, re.I):
        if not greeted:
            greet()
            greeted = True
    elif sentiment > 0:
        act_happy(sentiment)
    elif sentiment < 0:
        act_sad(sentiment)
    else:
        act_neutral()

def act_happy(sentiment):
    # act happy
    # the single flower bounces a little
    sound_file = "happy.wav"
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    # shake a little
    '''
    for i in range(3):
        REACTION_SERVO.angle = original_reaction_angle + 20
        time.sleep(0.1)
        REACTION_SERVO.angle = original_reaction_angle
        time.sleep(0.1)
     '''
    REACTION_SERVO.angle = original_reaction_angle
    #dance
    if sentiment > 0.4:
        for i in range(3):
            time.sleep(0.2)
            REACTION_SERVO.angle -= 10
            time.sleep(0.2)
            REACTION_SERVO.angle = original_reaction_angle
        
    # the bunch of flowers increases a little in height
    time.sleep(2)
    if CUMULATIVE_SERVO.angle <= int(cumulative_max_angle - cumulative_delta_angle*(1+sentiment)):
        CUMULATIVE_SERVO.angle += int(cumulative_delta_angle * (1 + sentiment))
    else: # the flower is already as happy as can be
        act_overlyhappy()

def act_neutral():
    # placeholder function in the weird situation where what's said is completely neutral
    REACTION_SERVO.angle = original_reaction_angle
        
def act_overlyhappy():
    # placeholder function for doing something when the flower is already as happy as can be
    shake_pot()
    sound_file = "dance.wav"
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    happy_lights()
    shake_pot()


def act_overlysad():
    # placeholder function for doing something when the flower is already as sad as can be
    # shake to get attention? speak?
    sound_file = "alarm.wav"
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    sad_lights()
    shake_pot()
        
def act_sad(score):
    # the single flower droops
    # the bunch of flowers decreases a little
    sound_file = "droop.wav"
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    # droop
    REACTION_SERVO.angle = (1 - abs(score))*original_reaction_angle
    '''
    end_angle = (1 - abs(score))*original_reaction_angle
    while REACTION_SERVO.angle > end_angle:
        REACTION_SERVO.angle -= 5
        time.sleep(0.1)
    '''    
    # Update cumulative bunch
    time.sleep(2)
    if CUMULATIVE_SERVO.angle >= cumulative_delta_angle:
        CUMULATIVE_SERVO.angle -= cumulative_delta_angle
    else: # the flower is already as sad as can be
        act_overlysad()
    time.sleep(2)
    REACTION_SERVO.angle = original_reaction_angle - 10
    
def greet():
    sound_file = "goodmorning.wav"
    pygame.mixer.music.load(sound_file)
    pygame.mixer.music.play()
    happy_lights()
    shake_pot()
    shake_pot()
    REACTION_SERVO.angle = original_reaction_angle
    CUMULATIVE_SERVO.angle = original_cumulative_angle
    POT_SERVO.angle = original_pot_angle

def happy_lights():
    # celebratory LED lights
    polarity = 0
    for j in range(40):
        for i in range(num_pixels):
            rc_index = (i * 256 // 10) + j*5
            pixels[i] = wheel(rc_index & 255)
            if (polarity == 1 and i % 2 == 0) or (polarity == 0 and i % 2 != 0):
                pixels[i] = OFF
        pixels.show()
        polarity = not polarity
        time.sleep(0.1)
    pixels.fill(OFF)

def sad_lights():
    for i in range(5):
        pixels.fill(BLUE)
        time.sleep(0.2)
        pixels.fill(OFF)
        time.sleep(0.2)
    pixels.fill(OFF)

def shake_pot():
    # shake the pot
    delay = 0.2
    shake_angle = 20
    POT_SERVO.angle += shake_angle
    for i in range(4):
        time.sleep(delay)
        POT_SERVO.angle -= 2*shake_angle
        time.sleep(delay)
        POT_SERVO.angle += 2*shake_angle
    POT_SERVO.angle = original_pot_angle
    
def wheel(pos):
    if pos < 0 or pos > 255:
        return (0,0,0)
    if pos < 85:
        return (255 - pos * 3, pos * 3, 0)
    if pos < 170:
        pos -= 85
        return (0, 255 - pos * 3, pos * 3)
    pos -= 170
    return (pos * 3, 0, 255 - pos * 3)

def repeat(transcript):
    t2s = gTTS('You said {}'.format(transcript), lang ='en')
    t2s.save('repeat.mp3')
    
    pygame.mixer.init()
    pygame.mixer.music.load('repeat.mp3')
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy(): 
        pygame.time.Clock().tick(10)
        
def main():
    # initialize servo angles
    REACTION_SERVO.angle = original_reaction_angle
    CUMULATIVE_SERVO.angle = original_cumulative_angle
    POT_SERVO.angle = original_pot_angle
    pixels.fill(OFF)
#    has_greeted = 0
    '''
    delta = 20
    for i in range(3):
        time.sleep(0.5)
        CUMULATIVE_SERVO.angle += delta
    time.sleep(2)
    CUMULATIVE_SERVO.angle = original_cumulative_angle
    time.sleep(2)
    for i in range(3):
        time.sleep(0.5)
        CUMULATIVE_SERVO.angle -= delta
    '''

    language_code = 'en-US'  # a BCP-47 language tag
    #time.sleep(3)
    #greet()
    #time.sleep(5)
    #act_happy(1)
    #time.sleep(2)
    #act_sad(-1)
    #time.sleep(2)
    #CUMULATIVE_SERVO.angle = 130
    #act_overlyhappy()
    #time.sleep(2)
    #CUMULATIVE_SERVO.angle = 0
    #act_sad(-1)
    #time.sleep(2)
    #act_happy(1)
    #time.sleep(1)
    #act_overlyhappy()
    
    #set up a client
    #make sure GCP is aware of the encoding, rate 
    config = types_speech.RecognitionConfig(
        encoding=enums_speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code)
    #our example uses streamingrecognition - most likely what you will want to use.
    #check out the simpler cases of asychronous recognition too!
    streaming_config = types_speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True)
    
    #this section is where the action happens:
    #a microphone stream is set up, requests are generated based on
    #how the audiofile is chunked, and they are sent to GCP using
    #the streaming_recognize() function for analysis. responses
    #contains the info you get back from the API. 
    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (types_speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        responses = client_s2t.streaming_recognize(streaming_config, requests)
        
        # Now, put the transcription responses to use.
        listen_print_loop(responses)


if __name__ == '__main__':
    main()