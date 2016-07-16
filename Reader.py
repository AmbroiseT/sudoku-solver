import cv2
import numpy as np
import copy
from DigitOCR import DigitOCR

class Reader:
    """
    Class which is used to read and analyse an image of as Sudoku Grid
    """
    def __init__(self):
        self.img = [[[]]]
        self.original = [[[]]]
        self.grid = [[[]]]
        self.rectifiedImage = [[[]]]
        self.largeurCorrected = 600

    def load_image(self, path):
        """
        Load image from path
        :param path:
        :return:
        """
        self.img = cv2.imread(path)
        self.original = copy.deepcopy(self.img)

    def clean_image(self):
        """
        Apply gaussian blur, adaptative threshold, and CLOSE to image to erase noise
        :return: None
        """
        self.img = cv2.GaussianBlur(self.img, (5,5),0)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img = cv2.adaptiveThreshold(self.img,  255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
        self.img = cv2.bitwise_not(self.img)

        matSize = int(self.img.shape[1]*0.01)
        matSize = 2
        kernel = np.ones((matSize,matSize), np.uint8)
        self.img = cv2.erode(self.img, kernel, iterations=1)
        self.img = cv2.dilate(self.img, kernel, iterations=3)

    def cut_image_from_clean(self):
        """
        cut the image into 81 cases
        :return:
        """
        assert (self.rectifiedImage is not None)
        tailleCase = self.largeurCorrected/9
        self.cases = [self.rectifiedImage[x*tailleCase:(x+1)*tailleCase, y*tailleCase:(y+1)*tailleCase] for x in range(9) for y in range(9)]

    def rectify_perspective(self):
        """
        Corrects the perspective of img
        :return:
        """
        contours, hierachy = cv2.findContours(copy.deepcopy(self.img), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        biggest = None
        max = -1
        for cnt in contours:
            ar = cv2.contourArea(cnt)
            if ar>max:
                max = ar
                biggest = cnt

        self.grid = np.zeros(self.img.shape, dtype = np.uint8)
        cv2.drawContours(self.grid, [biggest], -1, 255, 3)
        #cv2.imshow("Grid", self.grid)
        cv2.drawContours(self.original, [biggest], -1, (0,255,0), 3)

        corners = cv2.goodFeaturesToTrack(self.grid, 4, 0.01, 10)
        for corner in corners:
            cv2.circle(self.original, (corner[0][0], corner[0][1]), 5, (0,0,255))

        M = cv2.moments(biggest)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        sup_droit = (0,0)
        sup_gauche = (0,0)
        inf_droit = (0,0)
        inf_gauche = (0,0)

        for corner in corners:
            x, y = corner[0]
            if x<cX and y<cY:
                inf_gauche = (x,y)
            elif x<cX and y>cY:
                sup_gauche = (x,y)
            elif x>cX and y<cY:
                inf_droit = (x,y)
            elif x>cX and y>cY:
                sup_droit = (x,y)
        persp1 = np.float32([[inf_gauche[0],inf_gauche[1]],[inf_droit[0],inf_droit[1]],[sup_gauche[0],sup_gauche[1]],[sup_droit[0],sup_droit[1]]])
        persp2 = np.float32([[0,0],[self.largeurCorrected,0],[0,self.largeurCorrected],[self.largeurCorrected,self.largeurCorrected]])

        M = cv2.getPerspectiveTransform(persp1, persp2)

        self.rectifiedImage = cv2.warpPerspective(self.img, M, (self.largeurCorrected,self.largeurCorrected))

    def convert_to_matrix(self, trainer):
        """
        Convert the image to a 9x9 matrix of numbers
        :param trainer DigitOCM trainer. Assumed to be already trained
        :return:
        """
        assert isinstance(trainer, DigitOCR)
        matrix = np.zeros((9,9))
        for i in range(81):
            matrix[i/9][i%9] = trainer.read_image_tesseract(self.cases[i])
        return matrix

    def margin_cases(self):
        for i in range(len(self.cases)):
            case = self.cases[i]
            dim = case.shape[0]
            margin = dim*0.15
            minX = int(margin*2)
            maxX = int(dim - margin)
            minY = int(margin*2)
            maxY = int(dim-margin)
            case = case[minY:maxY,minX:maxX]
            self.cases[i] = case

    def show_image(self):
        '''
        Creates a windows to display img
        :return:
        '''
        cv2.imshow("Image", self.img)
        cv2.waitKey(0)

    def show_rectified(self):
        cv2.imshow("Image rectified", self.rectifiedImage)
        cv2.waitKey(0)

    def show_original(self):
        '''
        Creates a window to display original
        :return:
        '''
        cv2.imshow("Image originale", self.original)
        cv2.waitKey(0)
