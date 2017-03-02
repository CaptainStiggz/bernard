FFMPEG_BIN = "ffmpeg"

import cv2
import subprocess as sp
import numpy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as pyl
import os
import time
import signal
import pygame

print("starting...")

class AVStream:
   def __init__(self, path):
      self.path = path
      self.audio_opts = {
         'frequency': 44100,
         'channels': 2,
         'bps': 16
      }

      self.video_opts = {
         'width': 640,
         'height': 368,
         'fps': 25
      }

      # pygame for playing audio
      pygame.init()
      pygame.mixer.init(self.audio_opts['frequency'], -self.audio_opts['bps'], self.audio_opts['channels']) # ex: 44100 Hz, 16bit, 2 channels

   # print the stream info
   def get_stream_info(self):
      pipe = sp.Popen([FFMPEG_BIN,"-i", self.path, "-"], stdin=sp.PIPE, stdout=sp.PIPE,  stderr=sp.PIPE)
      pipe.stdout.readline()
      pipe.terminate()
      infos = pipe.stderr.read()
      print(infos)

   # get the video stream
   def open_video_stream(self, opts):
      command = [ FFMPEG_BIN,
         '-i', self.path,
         '-f', 'image2pipe',
         '-pix_fmt', 'rgb24',
         '-vcodec', 'rawvideo', 
         '-']
      return sp.Popen(command, stdout = sp.PIPE, stderr=sp.PIPE, stdin=open(os.devnull), bufsize=10**8)

   # get the audio stream
   def open_audio_stream(self):
      command = [ FFMPEG_BIN,
         '-i', self.path,
         '-f', 's%dle'%(self.audio_opts['bps']),
         '-vn',
         '-acodec', 'pcm_s%dle'%(self.audio_opts['bps']),
         '-ar', '%d'%(self.audio_opts['frequency']),
         '-ac', '%d'%(self.audio_opts['channels']),
         '-']
      return sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE, stdin=open(os.devnull), bufsize=10**8)

   # get the next video frame
   def next_video_frame(self, vpipe):
      w = self.video_opts['width']
      h = self.video_opts['height']

      # read next frame and throw away the data in the pipe's buffer.
      raw_image = vpipe.stdout.read(w*h*3)
      vpipe.stdout.flush()

      # transform the byte read into a numpy array
      return numpy.fromstring(raw_image, dtype='uint8')

   # process the video frame
   def process_video_frame(self, image):
      w = self.video_opts['width']
      h = self.video_opts['height']
      image = image.reshape((h, w, 3))
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      cv2.namedWindow("test", flags=cv2.WINDOW_NORMAL)
      cv2.imshow('test', gray)

   # get the next audio frame
   def next_audio_frame(self, apipe):
      audio_frequency = self.audio_opts['frequency']
      audio_channels = self.audio_opts['channels']
      audio_bits_per_sample = self.audio_opts['bps']
      duration = 1 / self.video_opts['fps']

      # convert bits/sample to bytes, mult by channels
      bytes_per_sample = int(audio_bits_per_sample / 8)
      frame_size = bytes_per_sample * audio_channels 

      # get the audio frame 
      data_length = audio_frequency * frame_size * duration
      print('exact: %f, rounded: %d'%(int(data_length), data_length))
      raw_audio = apipe.stdout.read(int(data_length))
      apipe.stdout.flush()

      # Reorganize raw_audio as a Numpy array with two-columns (1 per channel)
      return numpy.fromstring(raw_audio, dtype='int%d'%(2*audio_bits_per_sample))
      
   # process the audio frame
   def process_audio_frame(self, data, play):
      audio_channels = self.audio_opts['channels']
      data = data.reshape(int(len(data)/audio_channels), audio_channels)
      sound = pygame.sndarray.make_sound(data)
      sound.play()
      # if play:
      #    print("play")
      #    pygame.mixer.find_channel().play(sound)
      # else:
      #    print("queue")
      #    pygame.mixer.find_channel().queue(sound)

   # externally called run loop
   def run(self):
      vpipe = self.open_video_stream(self.video_opts)
      apipe = self.open_audio_stream()
      self.start_run_loop(vpipe, apipe)

   # the main run loop
   def start_run_loop(self, vpipe, apipe):
      c = 0
      sound_size = 0
      while True: 

         # get next video frame
         image = self.next_video_frame(vpipe)
         if(image.size == 0): 
            print("final sound size: %d"%(sound_size))
            break
         
         # process the video frame
         self.process_video_frame(image)
         
         # get the next audio frame
         data = self.next_audio_frame(apipe)
         self.process_audio_frame(data, c < 1)
         sound_size += data.size

         # exit
         if cv2.waitKey(1) & 0xFF == ord('q'):
            break

         # delay based on frame rate
         time.sleep(1/self.video_opts['fps'])
         c += 1

# read audio pipe samples for some duration
def read_audio_from_pipe(pipe, duration):
   audio_frequency = 44100
   audio_channels = 2
   audio_bits_per_sample = 16

   sound_size = 0
   bytes_per_sample = int(audio_bits_per_sample / 8)
   frame_size = bytes_per_sample * audio_channels # convert bits/sample to bytes, mult by channels

   # get the audio frame 
   raw_audio = pipe.stdout.read(audio_frequency * frame_size * duration)
   pipe.stdout.flush()

   # Reorganize raw_audio as a Numpy array with two-columns (1 per channel)
   audio_array = numpy.fromstring(raw_audio, dtype='int%d'%(2*audio_bits_per_sample))
   audio_array = audio_array.reshape(int(len(audio_array)/audio_channels), audio_channels)
   sound_size += audio_array.size

   sound = pygame.sndarray.make_sound(audio_array)
   sound.play()

   print("final sound size: %d"%(sound_size))
   

# file format
# file = 'rtmp://127.0.0.1:1935/live/ios'
stream = AVStream('recordings/sample.mp4')
stream.get_stream_info()
#apipe = stream.open_audio_stream()
#read_audio_from_pipe(apipe, 6)
stream.run()
time.sleep(3)

# end
cv2.destroyAllWindows()
print("done.")