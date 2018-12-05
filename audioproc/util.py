# coding: utf-8
import sys


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


def id(x):
    # 配列のメモリブロックアドレスを返す
    return x.__array_interface__['data'][0]
