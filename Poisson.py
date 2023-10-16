import cv2 as cv
import numpy as np

A = cv.imread('ocean.jpg')
B = cv.imread('astronomy.jpg')

# resize image
A = cv.resize(A, (800, 800), interpolation=cv.INTER_LINEAR)
B = cv.resize(B, (800, 800), interpolation=cv.INTER_LINEAR)

A = A[:, :int(800 * 0.6)]

mask = 255 * np.ones(A.shape, A.dtype)

# Poisson blending
ls_ = cv.seamlessClone(A, B, mask, (320, 400), cv.MIXED_CLONE)

cv.imwrite('Poisson_blending.jpg', ls_)

# bgr convert to rgb

ls_rgb = cv.cvtColor(ls_, cv.COLOR_BGR2RGB)

cv.imshow('Poisson blending', ls_rgb)

cv.waitKey(0)
cv.destroyAllWindows()
