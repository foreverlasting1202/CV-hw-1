import cv2 as cv
import numpy as np

A = cv.imread('../Dataset/ocean.jpg')
B = cv.imread('../Dataset/astronomy.jpg')

# resize image
A = cv.resize(A, (800, 800), interpolation=cv.INTER_LINEAR)
B = cv.resize(B, (800, 800), interpolation=cv.INTER_LINEAR)

n = 10

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(n):
    G = cv.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(n):
    G = cv.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[n - 1]]
for i in range(n - 1, 0, -1):
    GE = cv.pyrUp(gpA[i])
    w, h, _ = gpA[i - 1].shape
    GE = cv.resize(GE, (w, h))  # size keep same
    L = cv.subtract(gpA[i - 1], GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[n - 1]]
for i in range(n - 1, 0, -1):
    GE = cv.pyrUp(gpB[i])
    w, h, _ = gpB[i - 1].shape
    GE = cv.resize(GE, (w, h))  # size keep same
    L = cv.subtract(gpB[i - 1], GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, :int(cols * 0.6)], lb[:, int(cols * 0.6):]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in range(n // 2, n):
    ls_ = cv.pyrUp(ls_)
    w, h, _ = LS[i].shape
    ls_ = cv.resize(ls_, (w, h))  # size keep same
    ls_ = cv.add(ls_, LS[i])
# image with direct connecting each half

real = np.hstack((A[:, :int(800 * 0.6)], B[:, int(800 * 0.6):]))
cv.imwrite('../Result/Laplacian_blending.jpg', ls_)
cv.imwrite('../Result/Direct_blending.jpg', real)

# bgr convert to rgb

real_rgb = cv.cvtColor(real, cv.COLOR_BGR2RGB)
ls_rgb = cv.cvtColor(ls_, cv.COLOR_BGR2RGB)

cv.imshow('Direct blending', real_rgb)
cv.imshow('Laplacian blending', ls_rgb)

cv.waitKey(0)
cv.destroyAllWindows()
