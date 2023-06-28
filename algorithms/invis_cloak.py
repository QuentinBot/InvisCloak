import cv2
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import time
from . import Algorithm


class InvisCloak(Algorithm):
    """ init function """

    def __init__(self, n=10):
        self.past_imgs = []
        self.n = n
        self.bg_img = None

    """ Processes the input image"""

    def process(self, img):
        self.past_imgs.append(img)
        if len(self.past_imgs) > self.n:
            self.past_imgs.pop(0)

        """ 2.1 Vorverarbeitung """
        """ 2.1.1 Rauschreduktion """
        plotNoise = False  # Schaltet die Rauschvisualisierung ein
        if plotNoise:
            self._plotNoise(img, "Rauschen vor Korrektur")
        img = self._211_Rauschreduktion(img)
        if plotNoise:
            self._plotNoise(img, "Rauschen nach Korrektur")
        """ 2.1.2 HistogrammSpreizung """
        img = self._212_HistogrammSpreizung(img)

        """ 2.2 Farbanalyse """
        """ 2.2.1 RGB """
        # self._221_RGB(img)
        """ 2.2.2 HSV """
        # self._222_HSV(img)

        """ 2.3 Segmentierung und Bildmdifikation """
        img = self._23_SegmentUndBildmodifizierung(img)

        return img

    """ Reacts on mouse callbacks """

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            print("Mouse click!")
            #cur_img = self._211_Rauschreduktion()
            #self._221_RGB(cur_img)
            #self._222_HSV(cur_img)
            #maske, img = self._23_SegmentUndBildmodifizierung(cur_img)
            #cv2.imwrite(f"imgs/maske-{time.time()}.png", maske)

            #img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            #cv2.imwrite(f"imgs/bild-{time.time()}.png", img)

            self.bg_img = self._212_HistogrammSpreizung(self._211_Rauschreduktion(self.past_imgs[-1]))

    def _plotNoise(self, img, name: str):
        height, width = np.array(img.shape[:2])
        centY = (height / 2).astype(int)
        centX = (width / 2).astype(int)

        cutOut = 5
        tmpImg = deepcopy(img)
        tmpImg = tmpImg[centY - cutOut:centY + cutOut, centX - cutOut:centX + cutOut, :]

        outSize = 500
        tmpImg = cv2.resize(tmpImg, (outSize, outSize), interpolation=cv2.INTER_NEAREST)

        cv2.imshow(name, tmpImg)
        cv2.waitKey(1)

    def _211_Rauschreduktion(self, img=None):
        """
            Hier steht Ihr Code zu Aufgabe 2.1.1 (Rauschunterdrückung)
            - Implementierung Mittelwertbildung über N Frames
        """

        denoised_img = np.mean(np.array(self.past_imgs), axis=0).astype(np.uint8)
        return denoised_img

    def _212_HistogrammSpreizung(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.1.2 (Histogrammspreizung)
            - Transformation HSV
            - Histogrammspreizung berechnen
            - Transformation BGR
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 2] = (img[:, :, 2] - np.min(img[:, :, 2])) / (np.max(img[:, :, 2]) - np.min(img[:, :, 2])) * 255
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def _221_RGB(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.2.1 (RGB)
            - Histogrammberechnung und Analyse
        """

        for i in range(3):
            hist_size = 256
            hist_range = [0, 256]
            histr = cv2.calcHist([img], [i], None, [hist_size], hist_range)
            plt.plot(histr, color="b")
            plt.xlim([0, 256])
            plt.savefig(f"imgs/hist_rgb_no{i}-{time.time()}.png")
            plt.show()
            plt.clf()
        cv2.imwrite(f"imgs/cur_rgb_img{time.time()}.png", img)

    def _222_HSV(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.2.2 (HSV)
            - Histogrammberechnung und Analyse im HSV-Raum
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i in range(3):
            if i == 0:
                hist_range = [0, 180]
            else:
                hist_range = [0, 256]
            hist_size = hist_range[1]
            histr = cv2.calcHist([img], [i], None, [hist_size], hist_range)
            plt.plot(histr, color="b")
            plt.xlim(hist_range)
            plt.savefig(f"imgs/hist_hsv_no{i}-{time.time()}.png")
            plt.show()
            plt.clf()
        cv2.imwrite(f"imgs/cur_hsv_img{time.time()}.png", img)

    def _23_SegmentUndBildmodifizierung(self, img):
        """
            Hier steht Ihr Code zu Aufgabe 2.3.1 (StatischesSchwellwertverfahren)
            - Binärmaske erstellen
        """

        """
            Hier steht Ihr Code zu Aufgabe 2.3.2 (Binärmaske)
            - Binärmaske optimieren mit Opening/Closing
            - Wahl größte zusammenhängende Region
        """

        """
            Hier steht Ihr Code zu Aufgabe 2.3.1 (Bildmodifizerung)
            - Hintergrund mit Mausklick definieren
            - Ersetzen des Hintergrundes
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        channel1 = 0
        lower_bound0, upper_bound0 = 140, 190
        is_condition_0_true = (lower_bound0 < img[:, :, channel1]) * \
                              (img[:, :, channel1] < upper_bound0)
        lower_bound1, upper_bound1 = 0, 20
        is_condition_1_true = (lower_bound1 < img[:, :, channel1]) * \
                              (img[:, :, channel1] < upper_bound1)
        channel2 = 2
        lower_bound2, upper_bound2 = 0, 250
        is_condition_2_true = (lower_bound2 < img[:, :, channel2]) * \
                              (img[:, :, channel2] < upper_bound2)
        # binary_mask = (is_condition_0_true+is_condition_1_true)*is_condition_2_true
        binary_mask = is_condition_0_true * is_condition_2_true
        mask = binary_mask * 255
        mask = mask.astype(np.uint8)
        kernel_size = (15, 15)
        mask = cv2.dilate(mask, np.ones(kernel_size))
        mask = cv2.erode(mask, np.ones(kernel_size))
        mask = cv2.erode(mask, np.ones(kernel_size))
        mask = cv2.dilate(mask, np.ones(kernel_size))
        (cnts, _) = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts)==0:
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            return img
        c = max(cnts, key=cv2.contourArea)
        mask = cv2.drawContours(np.zeros_like(img, dtype=np.uint8), [c], -1, (1,1,1), thickness=-1).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        if self.bg_img is not None:
            fg = img * (1-mask)
            bg = self.bg_img * mask
            return fg+bg
        return img
