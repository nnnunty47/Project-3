# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
from project3 import get_raw_data
from project3 import data_clean

def test():
    city, raw_data = get_raw_data('AK Anchorage.pdf')
    clean_data = data_clean(raw_data)
    print(raw_data)
    print(clean_data)

if __name__ == '__main__':
    test()


