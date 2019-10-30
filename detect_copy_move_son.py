import cv2
import numpy as np
from PIL import Image
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt

'''
sift opencv nin her surumunde calismiyor. alttaki sekilde kutuphaneler yuklenmeli

$ pip uninstall opencv-python

$ pip uninstall opencv-contrib-python
then,
$ pip install opencv-contrib-python==3.4.2.16

$ pip install opencv-python==3.4.2.16

'''

img_gray = cv2.imread("im11_t.bmp", cv2.IMREAD_GRAYSCALE)
img_rgb = cv2.imread("im11_t.bmp", cv2.IMREAD_COLOR)

img_gray = cv2.resize(img_gray, (int(img_gray.shape[1] * 40 / 100), int(img_gray.shape[0] * 40 / 100)))
img_rgb = cv2.resize(img_rgb, (int(img_rgb.shape[1] * 40 / 100), int(img_rgb.shape[0] * 40 / 100)))
# print(img_rgb.shape)

# -----


segments = slic(img_rgb, n_segments=7, compactness=20, sigma=5, convert2lab=True)
regions = regionprops(segments, intensity_image=img_gray)


index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

for index, props in enumerate(regions):
    sift = cv2.xfeatures2d.SIFT_create()
    print(index)
    cx, cy = props.centroid  # cen
    print(props.intensity_image.shape)

    print(img_gray[props.intensity_image].shape)
    print(props.intensity_image)

    kp1, descriptors1 = sift.detectAndCompute(props.intensity_image, None)
    print(type(descriptors1))
    for index_ic, prop in enumerate(regions):
        if index == index_ic:
            continue
        sift_ic = cv2.xfeatures2d.SIFT_create()

        kp2, descriptors2 = sift_ic.detectAndCompute(prop.intensity_image, None)
        if np.array(descriptors1).size == 0 | np.array(descriptors2).size == 0:
            continue
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

        good_points = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                good_points.append(m)
                #cv2.circle(props.intensity_image, (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1])), 1, (255, 0, 255), 1)
                #cv2.circle(prop.intensity_image, (int(kp2[n.queryIdx].pt[0]), int(kp2[n.queryIdx].pt[1])), 1, (0, 255, 0), -1)  # eslesen objeyi isaretlemek icin

    cv2.imshow("Image " + str(index), props.intensity_image)
     # burada degisiklik olacak
    print("*-*-*-*-*-*-*-*-*-*-*-*--")

# ------

# show the output of SLIC
 # segments den gelen bilgilere gore matloplib de cizdik
fig = plt.figure("Superpixels -- %d segments" % (100))
ax = fig.add_subplot(1, 1, 1)
ax.imshow(mark_boundaries(img_rgb, segments, color=(0, 0, 0)))

plt.axis("off")

# show the plots
plt.show()

# ------

# sift = cv2.xfeatures2d.SIFT_create(900)
# keypoints_sift, descriptors = sift.detectAndCompute(img_gray, None)
                # print(descriptors.shape)
                # print(keypoints_sift[0].pt[0])
                # print(keypoints_sift[0].pt[1])
                # ------
                # kullanilan siniflandirma algoritmasinin parametrelerini ayarladik
                # index_params = dict(algorithm=0, trees=5)
                # search_params = dict()
                # flann = cv2.FlannBasedMatcher(index_params, search_params)

                # ---
                ### eslestirmeler burda yapilacak: iki bloga ait descriptors bilgisi parametre olarak verilir
                ### ve bunun sonucu olarak gelen matrisden distance lara gore  yakin olanlar secilir (ayni objedir)
                ### burasi her yapilan match den sonra calisacak
'''matches = flann.knnMatch(desc_1, desc_1, k=2)
                
                    good_points = []
                    for m, n in matches:
                        if m.distance < 0.6 * n.distance:
                            good_points.append(m)
                            #cv2.circle(img_rgb, (int(keypoints_sift[m.queryIdx].pt[0]), int(keypoints_sift[m.queryIdx].pt[1])), 4,
                                       (255, 0, 255),
                                       1) # eslesen objeyi isaretlemek icin
                            #cv2.circle(img_rgb, (int(keypoints_sift[n.queryIdx].pt[0]), int(keypoints_sift[n.queryIdx].pt[1])), 1,
                                       (0, 255, 0),
                                       -1) # eslesen objeyi isaretlemek icin
                            
                            '''

#img = cv2.drawKeypoints(img_rgb, keypoints_sift,  None)  # flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS :::: gosterimi degistir
# print(keypoints_sift)
# print(matches)
# print(good_points)

#cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
