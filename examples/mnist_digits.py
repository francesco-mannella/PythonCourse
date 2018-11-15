from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import mnist

im0 = mnist.test_images()[0]
plt.imshow(im0)
plt.show()
print(im0)

n_rows, _ = im0.shape
Row, Col = np.meshgrid(np.arange(n_rows), np.arange(n_rows))
im0sprs = np.vstack([Row.ravel(), Col.ravel(), im0.ravel()]).T
plt.scatter(im0sprs[:, 0], 255 - im0sprs[:, 1], s=im0sprs[:, 2]/10 )
plt.show()
print(im0sprs)
