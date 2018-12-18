# coding: utf-8
import sys
import time



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
            if progress != 0:
                elapsed_time = time.perf_counter() - self.start_time
                progress = percent - self.start_parcent
                remain_t =  (elapsed_time / progress) * (1 - percent)
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
        


def id(x):
    # 配列のメモリブロックアドレスを返す
    return x.__array_interface__['data'][0]
