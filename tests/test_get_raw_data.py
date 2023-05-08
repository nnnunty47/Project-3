# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
from project3 import get_raw_data

def test():
    city, raw_data = get_raw_data('AK Anchorage.pdf')
    print(city)
    print(len(raw_data))

if __name__ == '__main__':
    test()
