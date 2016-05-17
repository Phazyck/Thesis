import sys
from time import time

class ProgressBar(object):

    def init(self):
        self.last_bar = -1
        self.count = 0
        self.d = 0
        self.t0 = time()
        self.start = time()

    def __init__(self, width, max):
        self.width = width
        self.max = max
        
        smax = str(max)
        digits = str(len(smax))
        self.fmt = "%" + digits + "d of " + smax + " (%6.2f%%) [%s%s] %s left\r" 
        
        self.init()
        
    def get_time_estimate(self):
        time_a = self.start
        time_b = time()
        
        time_d = time_b - time_a
        
        total_time = (time_d * float(self.max)) / float(self.count)
        time_left = total_time - time_d
        
        time_m, time_s = divmod(time_left, 60)
        time_h, time_m = divmod(time_m, 60)
        
        return "%2dh%2dm%5.2fs" % (int(time_h), int(time_m), time_s)

    def post(self):
        
        self.count += 1
    
        t1 = time()
        diff = (t1 - self.t0)
        self.d += diff
        self.t0 = t1 
    
        if self.d > 0.1:
            self.d -= 0.1
            
            a = self.count
            b = self.max
            
            pct = (float(a) / float(b))
            
            bar_fill = int(pct * self.width)
            
            estimate = self.get_time_estimate()
        
            bar_empty = self.width - bar_fill
            print self.fmt % (a, 100.0 * pct, "=" * bar_fill, " " * bar_empty, estimate),
            sys.stdout.flush()
            
    def end(self):
        estimate = self.get_time_estimate()
        print self.fmt % (self.max, 100.0, "=" * self.width, "", estimate),
        print
        self.init()