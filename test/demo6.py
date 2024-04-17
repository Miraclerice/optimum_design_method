# -*- coding: utf-8 -*-
# @Author: MiracleRice
# Blog   : miraclerice.com
import datetime
import time

import numpy as np
import matplotlib.pyplot as plt

img = np.random.randn(10, 10)

# fig = plt.imshow(img)

fig, ax = plt.subplots()
# 去掉右边框和上边框
# ax.imshow(img)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

# 去掉右边框和上边框
ax.imshow(img, cmap='Greys')
ax = plt.gca()  # get current axis
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')


# 去除图例边框
# plt.legend(frameon=False)

# plt.axis('off')  # 一次性去掉所有框

# plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
#                     hspace=0, wspace=0)
# plt.margins(0, 0)
# plt.savefig('image3.png', bbox_inches='tight', pad_inches=0.1) # 默认pad_inches=0.1
# plt.savefig('image3.png', bbox_inches='tight', pad_inches=0)
# plt.show()
"""
savefig(fname, *, transparent=None, dpi='figure', format=None,
        metadata=None, bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto', backend=None,
        **kwargs
       )
"""

print(time.time())
# print(datetime.datetime.now())
# print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))