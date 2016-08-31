import network
import os
import cv2


theta = network.load_theta()

for f in os.listdir("test-data"):
	img = cv2.imread("test-data/"+f, cv2.IMREAD_GRAYSCALE)
	res = network.predict_from_image(img, theta)
	cv2.imshow("Image", img)
	print res
	cv2.waitKey(0)

