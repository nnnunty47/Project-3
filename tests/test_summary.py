# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
from project3 import summarize

def test():
    raw_data = 'Test, test, and testing.'
    keywords, summary = summarize(raw_data)
    print(keywords)
    print(summary)

if __name__ == '__main__':
    test()
