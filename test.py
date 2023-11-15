# import numpy as np
#
# a = np.empty(10)
#
# # for i in range(0, 10, 1):
# #     a[i] = i
# print(a)

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

N = 5

x = np.random.rand(N)
y = np.random.rand(N)

plt.plot(x, y, 'r*')

for xy in zip(x, y):
   plt.annotate('(%.2f, %.2f)' % xy, xy=xy)

plt.show()