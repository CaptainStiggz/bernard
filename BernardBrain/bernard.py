from brains.avstream import AVStream
import time
import cv2
import pygame

# bernard stream
class BernardStream(AVStream):
   
   # pygame for playing audio
   def before_run(self):
      pygame.init()
      pygame.mixer.init(self.audio_opts['frequency'], -self.audio_opts['bits_per_sample'], self.audio_opts['channels']) # ex: 44100 Hz, 16bit, 2 channels

   # process the video frame
   def process_video_frame(self, image):
      w = self.video_opts['width']
      h = self.video_opts['height']
      image = image.reshape((h, w, 3))
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      cv2.namedWindow("test", flags=cv2.WINDOW_NORMAL)
      cv2.imshow('test', gray)

   # process the audio frame
   def process_audio_frame(self, data):
      audio_channels = self.audio_opts['channels']
      data = data.reshape(int(len(data)/audio_channels), audio_channels)
      sound = pygame.sndarray.make_sound(data)
      # print("sound length: %f"%(sound.get_length()))
      sound.play()
      # if play:
      #    print("play")
      #    pygame.mixer.find_channel().play(sound)
      # else:
      #    print("queue")
      #    pygame.mixer.find_channel().queue(sound)

# run the program
# file = 'rtmp://127.0.0.1:1935/live/ios'
stream = BernardStream('recordings/sample.mp4')
stream.run()
time.sleep(3)

# end
cv2.destroyAllWindows()
print("done.")