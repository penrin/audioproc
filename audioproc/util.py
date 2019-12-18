# coding: utf-8
import sys
import time
from math import ceil
import threading


def progressbar(percent, end=1, bar_length=40, slug='#', space='-'):
    percent = percent / end # float
    slugs = slug * int( round( percent * bar_length ) )
    spaces = space * ( bar_length - len( slugs ) )
    bar = slugs + spaces
    sys.stdout.write("\r[{bar}] {percent:.1f}% ".format(
    	bar=bar, percent=percent*100.
    ))
    sys.stdout.flush()
    if percent == 1:
        print()



class ProgressBar():

    def __init__(self, bar_length=40, slug='#', space='-', countdown=True):

        self.bar_length = bar_length
        self.slug = slug
        self.space = space
        self.countdown = countdown
        self.start_time = None
        self.start_parcent = 0
    
    def bar(self, percent, end=1, tail=''):
        percent = percent / end

        if self.countdown == True:
            progress = percent - self.start_parcent
            
            if self.start_time == None:
                self.start_time = time.perf_counter()
                self.start_parcent = percent
                remain = 'Remain --:--:--'
                
            elif progress == 0:
                remain = 'Remain --:--:--'
            
            elif progress != 0:
                elapsed_time = time.perf_counter() - self.start_time
                progress = percent - self.start_parcent
                remain_t = (elapsed_time / progress) * (1 - percent)
                remain_t = ceil(remain_t)
                h = remain_t // 3600
                m = remain_t % 3600 // 60
                s = remain_t % 60
                remain = 'Remain %02d:%02d:%02d' % (h, m, s) 
                
        else:
            remain = ''
        
        len_slugs = int(percent * self.bar_length)
        slugs = self.slug * len_slugs
        spaces = self.space * (self.bar_length - len_slugs)
        txt = '\r[{bar}] {percent:.1%} {remain} {tail}'.format(
                bar=(slugs + spaces), percent=percent,
                remain=remain, tail=tail)
        if percent == 1:
            txt += '\n'
            self.start_time = None
        sys.stdout.write(txt)
        sys.stdout.flush()
        

class ProgressBar2(ProgressBar):

    def __init__(self, end, bar_length=40, slug='#', space='-', countdown=True):
        
        super().__init__(bar_length=40, slug='#', space='-', countdown=True)
        self.counter = 0
        self.end = end
        self.bar()
        
    def bar(self, tail=''):
        super().bar(self.counter, end=self.end, tail=tail)
        self.counter += 1
        

        
class Propeller:
    
    def __init__(self, charlist=None, sleep=0.1):
        if charlist is None:
            self.charlist = ['|', '/', '-', '\\']
        self.sleep = sleep
        self.working = True    

    def progress(self):
        N = len(self.charlist)
        i = 0
        sys.stdout.write(' ')
        while self.working:
            sys.stdout.write('\b' + self.charlist[i])
            sys.stdout.flush()
            time.sleep(self.sleep)
            i = (i + 1) % N
        sys.stdout.write('\b' + 'done')
        sys.stdout.flush()
        
    def start(self):
        self.working = True
        self.thread = threading.Thread(target=self.progress)
        self.thread.start()
        
    def end(self):
        self.working = False
        self.thread.join()
        


def id(x):
    # 配列のメモリブロックアドレスを返す
    return x.__array_interface__['data'][0]





if __name__ == '__main__':
    
    N = 100
    pg = ProgressBar2(N)
    for n in range(N):    
        time.sleep(0.02)
        pg.bar()
    
    print()
    print('Propeller...', end='')
    p = Propeller()
    p.start()
    time.sleep(3)
    p.end()
    
        
