import subprocess as sp
import cv2
import numpy
import os
import datetime, threading, time
from .ffprobe import FFProbe

FFMPEG_BIN = "ffmpeg"

class AVStream:
   def __init__(self, path):
      self.path = path
      self.audio_opts = {
         'frequency': 44100,
         'channels': 2,
         'bits_per_sample': 16
      }

      self.video_opts = {
         'width': 640,
         'height': 368,
         'fps': 30
      }
      
      self.loop_time = 0

   # print the stream info
   def update_stream_metadata(self):
      probe = FFProbe(self.path)

      # assumes 2 streams, one for video one for audio
      for stream in probe.streams:

         # store video dimensions and fps
         if stream.isVideo():
            fps = stream.frames() / stream.durationSeconds()
            size = stream.frameSize()
            print("VIDEO: %dx%d, %d fps"%(size[0], size[1], fps))
            self.video_opts['width'] = size[0]
            self.video_opts['height'] = size[1]
            self.video_opts['fps'] = fps

         # don't really do much, just print settings
         elif stream.isAudio():
            af = self.audio_opts['frequency']
            ac = self.audio_opts['channels']
            abps = self.audio_opts['bits_per_sample']
            print("AUDIO: %d Hz, %d-channel, %d bits/sample"%(af, ac, abps))

   # get the video stream
   def open_video_stream(self):
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
         '-f', 's%dle'%(self.audio_opts['bits_per_sample']),
         '-vn',
         '-acodec', 'pcm_s%dle'%(self.audio_opts['bits_per_sample']),
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
      print("WARNING: process_video_frame() not implemented")

   # get the next audio frame
   def next_audio_frame(self, apipe):
      audio_frequency = self.audio_opts['frequency']
      audio_channels = self.audio_opts['channels']
      audio_bits_per_sample = self.audio_opts['bits_per_sample']
      duration = 1 / self.video_opts['fps']

      # convert bits/sample to bytes, mult by channels
      bytes_per_sample = int(audio_bits_per_sample / 8)
      frame_size = bytes_per_sample * audio_channels 

      # get the audio frame 
      data_length = audio_frequency * frame_size * duration
      # print('exact: %f, rounded: %d'%(int(data_length), data_length))
      raw_audio = apipe.stdout.read(int(data_length))
      apipe.stdout.flush()

      # Reorganize raw_audio as a Numpy array with two-columns (1 per channel)
      return numpy.fromstring(raw_audio, dtype='int%d'%(2*audio_bits_per_sample))
      
   # process the audio frame
   def process_audio_frame(self, data):
      print("WARNING: process_audio_frame() not implemented")


   # called after metadata is updated, before streams are initialized
   def before_run(self):
      print("SKIPPING: before_run()")

   # externally called run loop
   def run(self):
      self.update_stream_metadata()
      self.before_run()
      vpipe = self.open_video_stream()
      apipe = self.open_audio_stream()
      self.start_run_loop(vpipe, apipe)

      # # put periodic frame grabbing on a backgrond thread
      # timerThread = threading.Thread(target=self.start_run_loop, args=(vpipe, apipe))
      # timerThread.daemon = True
      # timerThread.start()
      # while True:
      #    pass

   # the main run loop
   def start_run_loop(self, vpipe, apipe):
      sleep_delay = 1 / self.video_opts['fps']

      # run a loop pass and wait
      while self.loop_pass(vpipe, apipe): 
         if cv2.waitKey(1) & 0xFF == ord('q'): break
         time.sleep(sleep_delay)
         
      self.close(vpipe, apipe)

      # self.loop_time = time.time()
      # while True:
      #    self.loop_pass(vpipe, apipe)
      #    self.loop_time = self.loop_time + (1 / self.video_opts['fps'])
      #    time.sleep(self.loop_time - time.time())

   # single pass of the run loop
   def loop_pass(self, vpipe, apipe):

      # get next video frame
      image = self.next_video_frame(vpipe)
      if(image.size == 0): 
         return False
      
      # process the video frame
      self.process_video_frame(image)
      
      # get the next audio frame
      data = self.next_audio_frame(apipe)
      if(data.size != 0):
         self.process_audio_frame(data)

      return True

   # called after each loop pass
   def on_loop_pass(self, image, audio):
      print("SKIPPING: on_loop_pass()")

   # cleanup and close
   def close(self, vpipe, apipe):
      vpipe.terminate()
      apipe.terminate()
