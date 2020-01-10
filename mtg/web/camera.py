import cv2
import threading
import time
from collections import deque


thread = None


class Camera(object):
    """
    
    """
    def __init__(self, fps=20, max_frames=None, video_source=0):
        """
        Args:
            None

        Kwargs:
            fps <int> - frames-per-second
            max_frames <int> - maximum of recorded frames
            video_source <int/str> - video source channel
        """
        self.fps = fps
        self.video_source = video_source
        self.camera = cv2.VideoCapture(self.video_source)
        self.max_frames = int(5*self.fps) if max_frames is None else int(max_frames)
        self.frames = deque([])
        self.isrecording = False
    
    def run(self, verbose=True):
        """
        Start the capture loop

        Args:
            None

        Kwargs:
            verbose <bool> - print statements to the command line

        Return:
            None
        """
        global thread
        if thread is None:
            thread = threading.Thread(target=self._capture_loop)
            if verbose:
                print("Starting thread...")
            thread.start()
            self.isrecording = True

    def _capture_loop(self, verbose=True):
        """
        Start the capture loop

        Args:
            None

        Kwargs:
            verbose <bool> - print statements to the command line

        Return:
            None
        """
        spf = 1./self.fps
        if verbose:
            print("Observing...")
        while self.isrecording:
            v, im = self.camera.read()
            if v:
                if len(self.frames) >= self.max_frames:
                    self.frames.popleft()
                self.frames.append(im)
            time.sleep(spf)

    def stop(self):
        """
        Stop recording
        """
        self.isrunning = False

    def get_frame(self, bytes=True):
        """
        Retrieve last recorded frame

        Args:
            None

        Kwargs:
            bytes <bool> - return byte-encoded image

        Return:
            img <> - TODO
        """
        if len(self.frames) > 0:
            if bytes:
                img = cv2.imencode('.png', self.frames[-1])[1].tobytes()
            else:
                img = self.frames.pop()
        else:
            # with open("templates/404.jpeg","rb") as f:
            #     img = f.read()
            return None
        return img

    
