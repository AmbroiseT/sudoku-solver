import cv2
import numpy as np
import random
import copy
import Reader
import Image
from pytesseract import image_to_string

class DigitOCR:
    stockDim = 100
    def __init__(self):
        self.model = cv2.KNearest()

    def save_training_data(self, outputSamples="training/samples.data", outputResponses="training/responses.data"):
        responses_to_save = np.array(self.responses, np.float32)
        responses_to_save = responses_to_save.reshape((responses_to_save.size, 1))

        np.savetxt(outputSamples, self.samples)
        np.savetxt(outputResponses, responses_to_save)

    def create_training_data(self, outputSamples="training/samples.data", outputResponses="training/responses.data"):
        '''
        Train on generated data and creates files containing the data
        :param outputSamples:
        :param outputResponses:
        :return:
        '''
        dimCase = self.stockDim

        self.samples = np.empty((0,self.stockDim**2))
        self.responses = []

        fonts = [cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_ITALIC][0:1]

        for iterator in range(2):
            for font in fonts:
                for i in range(0,10):

                    if i!=0:
                        img = np.zeros((dimCase*2, dimCase*2, 1), np.uint8)
                        cv2.putText(img, str(i), (random.randint(int(0.4*self.stockDim), int(0.7*self.stockDim)),random.randint(int(1.1*self.stockDim), int(1.5*self.stockDim))),
                                    font, 0.05*self.stockDim, 255,5)

                        #To improve:
                        contours,hierarchy = cv2.findContours(copy.deepcopy(img),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                        cnt = contours[0]
                        x,y,w,h = cv2.boundingRect(cnt)
                        img = img[y:y+h,x:x+w]
                    else:
                        img = np.zeros((dimCase, dimCase, 1), np.uint8)

                    img = cv2.resize(img, (self.stockDim,self.stockDim))

                    self.responses.append(i)
                    sample = img.reshape((1,self.stockDim**2))
                    self.samples = np.append(self.samples, sample, 0)

        self.save_training_data()

    def train_from_grid(self, outputSamples="training/samples.data", outputResponses="training/responses.data"):
        dimCase = self.stockDim
        self.samples = np.empty((0,self.stockDim**2))
        self.responses = []
        for i in range(6, 10):
            reader = Reader.Reader()
            reader.load_image("data/sudokus/sudoku"+str(i)+".png")
            reader.clean_image()
            reader.rectify_perspective()
            reader.cut_image_from_clean()
            for img in reader.cases:
                contours,hierarchy = cv2.findContours(copy.deepcopy(img),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                maxArea = -1
                biggest = None
                for ctn in contours:
                    area = cv2.arcLength(ctn, True)
                    if area>maxArea and area > 25:
                        maxArea = area
                        biggest = ctn
                x,y,w,h = cv2.boundingRect(biggest)
                img = img[y:y+h,x:x+w]

                img = cv2.resize(img, (self.stockDim,self.stockDim))

                img = img.reshape((1, img.size))
                img = np.float32(img)

                self.responses.append(i)
                self.samples = np.append(self.samples, img, 0)




    def train_from_images(self, outputSamples="training/samples.data", outputResponses="training/responses.data"):
        dimCase = self.stockDim
        samples = np.empty((0,self.stockDim**2))
        responses = []

        for i in range(0, 10):
            img = cv2.imread("data/hand/mnist_train"+str(i)+".jpg", cv2.IMREAD_GRAYSCALE)

            blur = cv2.GaussianBlur(img, (5, 5), 0)

            #thresh = cv2.adaptiveThreshold(blur, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            ret, thresh = cv2.threshold(blur, 127, 155, cv2.THRESH_BINARY)

            kernel = np.ones((2,2),np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            contours, hierarchy = cv2.findContours(copy.deepcopy(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


            for cnt in contours:
                if cv2.contourArea(cnt)>25:
                    [x,y,w,h] = cv2.boundingRect(cnt)

                    roi = img[y:y+h,x:x+w]
                    roi = cv2.resize(roi, (self.stockDim,self.stockDim))
                    responses.append(i)
                    sample = roi.reshape((1,self.stockDim**2))
                    samples = np.append(samples, sample, 0)

            self.save_training_data()

    def train_from_caps(self, outputSamples="training/samples.data", outputResponses="training/responses.data"):
        dimCase = self.stockDim
        samples = np.empty((0,self.stockDim**2))
        responses = []

        for i in range(1, 10):
            img = cv2.imread("data/word/"+str(i)+".PNG", cv2.IMREAD_GRAYSCALE)

            contours, hierarchy = cv2.findContours(copy.deepcopy(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt)>25:
                    [x,y,w,h] = cv2.boundingRect(cnt)

                    roi = img[y:y+h,x:x+w]
                    roi = cv2.resize(roi, (self.stockDim, self.stockDim))
                    responses.append(i)
                    sample = roi.reshape((1, self.stockDim**2))
                    samples = np.append(samples, sample, 0)

        print "found "+str(len(responses))+" contours"
        self.save_training_data()

    def load_training_data(self, inputSamples="training/samples.data", inputResponses="training/responses.data"):
        """
        Load training data from files and train the model with the data
        :param inputSamples:
        :param inputResponses:
        :return:
        """
        samples = np.loadtxt(inputSamples, np.float32)
        responses = np.loadtxt(inputResponses, np.float32)

        responses = responses.reshape((responses.size, 1))
        self.model.train(samples, responses)

    def train(self):
        responses_to_train = np.array(self.responses, np.float32)
        responses_to_train = responses_to_train.reshape((responses_to_train.size, 1))
        samples_to_train = np.array(self.samples, np.float32)
        self.model.train(samples_to_train, responses_to_train)

    def add_training(self, sample, response):
        self.responses.append(response)
        roi = cv2.resize(sample, (self.stockDim, self.stockDim))
        sample = roi.reshape((1, self.stockDim**2))
        self.samples = np.append(self.samples, sample, 0)

    def read_image_tesseract(self,img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(img)
        
        result = image_to_string(pil_img, config="-psm 10 -c tessedit_char_whitelist=123456789")
        print result
        try:
            return int(result)
        except ValueError:
            return 0

    def read_image(self, img):
        percent = (cv2.countNonZero(img)*100.0)/img.size
        if percent<25:
            return 0

        contours,hierarchy = cv2.findContours(copy.deepcopy(img),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        maxArea = -1
        biggest = None
        for ctn in contours:
            area = cv2.arcLength(ctn, True)
            if area>maxArea and area>25:
                maxArea = area
                biggest = ctn

        if biggest is not None:
            x,y,w,h = cv2.boundingRect(biggest)
            img = img[y:y+h,x:x+w]
        else:
            return 0

        img = cv2.resize(img, (self.stockDim,self.stockDim))

        img = img.reshape((1, img.size))
        img = np.float32(img)
        ret, results, neigh_resp, dists = self.model.find_nearest(img, k=1)
        return results[0]

