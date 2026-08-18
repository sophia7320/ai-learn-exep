import time

class Timer:
    def __init__(self):
        self.times = []

    def start(self):
        self.tick = time.time()

    def stop(self):
        self.times.append(time.time() - self.tick)
        return self.times[-1]