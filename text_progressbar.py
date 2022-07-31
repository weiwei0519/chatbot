# coding=UTF-8
# 
# 

'''
@File: text_progressbar.py
@Author: Wei Wei
@Time: 2022/7/31 17:09
@Email: weiwei_519@outlook.com
@Software: PyCharm
'''

import time
import logging
import progressbar

progressbar.streams.wrap_stderr()
logging.basicConfig()
progressbar.ProgressBar.setIndeterminateDrawable()
for i in progressbar.progressbar(range(10)):
    logging.error('Got %d', i)
    time.sleep(0.2)
