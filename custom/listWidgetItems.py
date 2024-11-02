import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.filters
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtWidgets import QListWidgetItem, QPushButton, QFileDialog
from flags import *
from skimage import filters, restoration


class MyItem(QListWidgetItem):
    def __init__(self, name=None, parent=None):
        super(MyItem, self).__init__(name, parent=parent)
        self.setIcon(QIcon('icons/color.png'))
        self.setSizeHint(QSize(60, 60))  # size

    def get_params(self):
        protected = [v for v in dir(self) if v.startswith('_') and not v.startswith('__')]
        param = {}
        for v in protected:
            param[v.replace('_', '', 1)] = self.__getattribute__(v)
        return param

    def update_params(self, param):
        for k, v in param.items():
            if '_' + k in dir(self):
                self.__setattr__('_' + k, v)


class GrayingItem(MyItem):
    def __init__(self, parent=None):
        super(GrayingItem, self).__init__('灰度化', parent=parent)
        self._mode = BGR2GRAY_COLOR

    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img


class FilterItem(MyItem):

    def __init__(self, parent=None):
        super().__init__('滤波器', parent=parent)
        self._ksize = 3
        self._kind = MEAN_FILTER
        self._sigmax = 0
        self._cutoff = 0.01
        self._cutoffSize = 10
        self._notch_x = 0
        self._notch_y = 0

    def __call__(self, img):
        if self._kind == MEAN_FILTER:
            img = cv2.blur(img, (self._ksize, self._ksize))
        elif self._kind == GAUSSIAN_FILTER:
            img = cv2.GaussianBlur(img, (self._ksize, self._ksize), self._sigmax)
        elif self._kind == MEDIAN_FILTER:
            img = cv2.medianBlur(img, self._ksize)
        elif self._kind == BOX_FILTER:
            img = cv2.boxFilter(img, -1, (self._ksize, self._ksize))
        elif self._kind == LOW_PASS:
            img = img.astype(np.float32) / 255.0
            img = filters.butterworth(img, self._cutoff, False)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        elif self._kind == HIGH_PASS:
            img = img.astype(np.float32) / 255.0
            img = filters.butterworth(img, self._cutoff, True)
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        elif self._kind == LAPLACIAN_ENHANCE:
            kernel1 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
            laplacian = cv2.filter2D(img, cv2.CV_64F, kernel1)
            laplacian = cv2.convertScaleAbs(laplacian)
            sharpened = cv2.addWeighted(img, 1.5, laplacian, -0.5, 0)
            img = sharpened
        elif self._kind == HOMO_FILTER:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.float64(gray)
            rows, cols = gray.shape
            r1 = 0.5
            rh = 2
            c = 4
            h = 2.0
            l = 0.5
            gray_fft = np.fft.fft2(gray)
            gray_fftshift = np.fft.fftshift(gray_fft)
            dst_fftshift = np.zeros_like(gray_fftshift)
            M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows // 2, rows // 2))
            D = np.sqrt(M ** 2 + N ** 2)
            Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / self._cutoffSize ** 2))) + r1
            dst_fftshift = Z * gray_fftshift
            dst_fftshift = (h - l) * dst_fftshift + l
            dst_ifftshift = np.fft.ifftshift(dst_fftshift)
            dst_ifft = np.fft.ifft2(dst_ifftshift)
            dst = np.real(dst_ifft)
            dst = np.uint8(np.clip(dst, 0, 255))
            img = dst
        elif self._kind == NOTCH_FILTER:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = np.float64(gray)
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            notch_centers = [(ccol + self._notch_x, crow + self._notch_y), (ccol - self._notch_x, crow - self._notch_y)]
            mask = np.ones((rows, cols), np.float32)
            for center in notch_centers:
                cv2.circle(mask, center, self._cutoffSize, 0, thickness=-1)
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            fshift_filtered = fshift * mask
            img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
            dst = np.real(img_back)
            dst = np.uint8(np.clip(dst, 0, 255))
            img = dst

        return img


class MorphItem(MyItem):
    def __init__(self, parent=None):
        super().__init__(' 形态学 ', parent=parent)
        self._ksize = 3
        self._op = ERODE_MORPH_OP
        self._kshape = RECT_MORPH_SHAPE

    def __call__(self, img):
        op = MORPH_OP[self._op]
        kshape = MORPH_SHAPE[self._kshape]
        kernal = cv2.getStructuringElement(kshape, (self._ksize, self._ksize))
        img = cv2.morphologyEx(img, self._op, kernal)
        return img


class GradItem(MyItem):
    def __init__(self, parent=None):
        super().__init__('梯度', parent=parent)
        self._kind = SOBEL_GRAD
        self._ksize = 3
        self._dx = 1
        self._dy = 0

    def __call__(self, img):
        if self._dx == 0 and self._dy == 0 and self._kind != LAPLACIAN_GRAD:
            self.setBackground(QColor(255, 0, 0))
            self.setText('图像梯度 （无效: dx与dy不同时为0）')
        else:
            self.setBackground(QColor(200, 200, 200))
            self.setText('图像梯度')
            if self._kind == SOBEL_GRAD:
                img = cv2.Sobel(img, -1, self._dx, self._dy, self._ksize)
            elif self._kind == SCHARR_GRAD:
                img = cv2.Scharr(img, -1, self._dx, self._dy)
            elif self._kind == LAPLACIAN_GRAD:
                img = cv2.Laplacian(img, -1)
        return img


class ThresholdItem(MyItem):
    def __init__(self, parent=None):
        super().__init__('阈值处理', parent=parent)
        self._thresh = 127
        self._maxval = 255
        self._method = BINARY_THRESH_METHOD

    def __call__(self, img):
        method = THRESH_METHOD[self._method]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.threshold(img, self._thresh, self._thresh, method)[1]
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img


class EdgeItem(MyItem):
    def __init__(self, parent=None):
        super(EdgeItem, self).__init__('边缘检测', parent=parent)
        self._thresh1 = 20
        self._thresh2 = 100

    def __call__(self, img):
        img = cv2.Canny(img, threshold1=self._thresh1, threshold2=self._thresh2)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img


class ContourItem(MyItem):
    def __init__(self, parent=None):
        super(ContourItem, self).__init__('轮廓检测', parent=parent)
        self._mode = TREE_CONTOUR_MODE
        self._method = SIMPLE_CONTOUR_METHOD
        self._bbox = NORMAL_CONTOUR

    def __call__(self, img):
        mode = CONTOUR_MODE[self._mode]
        method = CONTOUR_METHOD[self._method]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cnts, _ = cv2.findContours(img, mode, method)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if self._bbox == RECT_CONTOUR:
            bboxs = [cv2.boundingRect(cnt) for cnt in cnts]
            print(bboxs)
            for x, y, w, h in bboxs:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
        elif self._bbox == MINRECT_CONTOUR:
            bboxs = [np.int0(cv2.boxPoints(cv2.minAreaRect(cnt))) for cnt in cnts]
            img = cv2.drawContours(img, bboxs, -1, (255, 0, 0), thickness=2)
        elif self._bbox == MINCIRCLE_CONTOUR:
            circles = [cv2.minEnclosingCircle(cnt) for cnt in cnts]
            print(circles)
            for (x, y), r in circles:
                img = cv2.circle(img, (int(x), int(y)), int(r), (255, 0, 0), thickness=2)
        elif self._bbox == NORMAL_CONTOUR:
            img = cv2.drawContours(img, cnts, -1, (255, 0, 0), thickness=2)

        return img


class EqualizeItem(MyItem):
    def __init__(self, parent=None):
        super().__init__('均衡化', parent=parent)
        self._blue = True
        self._green = True
        self._red = True

    def __call__(self, img):
        b, g, r = cv2.split(img)
        if self._blue:
            b = cv2.equalizeHist(b)
        if self._green:
            g = cv2.equalizeHist(g)
        if self._red:
            r = cv2.equalizeHist(r)
        return cv2.merge((b, g, r))


class HoughLineItem(MyItem):
    def __init__(self, parent=None):
        super(HoughLineItem, self).__init__('直线检测', parent=parent)
        self._rho = 1
        self._theta = np.pi / 180
        self._thresh = 10
        self._min_length = 20
        self._max_gap = 5

    def __call__(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lines = cv2.HoughLinesP(img, self._rho, self._theta, self._thresh, minLineLength=self._min_length,
                                maxLineGap=self._max_gap)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if lines is None: return img
        for line in lines:
            for x1, y1, x2, y2 in line:
                img = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        return img


class LightItem(MyItem):
    def __init__(self, parent=None):
        super(LightItem, self).__init__('亮度调节', parent=parent)
        self._alpha = 1
        self._beta = 0

    def __call__(self, img):
        blank = np.zeros(img.shape, img.dtype)
        img = cv2.addWeighted(img, self._alpha, blank, 1 - self._alpha, self._beta)
        return img


class GammaItem(MyItem):
    def __init__(self, parent=None):
        super(GammaItem, self).__init__('伽马校正', parent=parent)
        self._gamma = 1

    def __call__(self, img):
        gamma_table = [np.power(x / 255.0, self._gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        return cv2.LUT(img, gamma_table)


class MatchingItem(MyItem):
    def __init__(self, parent=None):
        super(MatchingItem, self).__init__('直方图匹配', parent=parent)
        self._refImg = None
        self._pushed1 = False
        self._pushed2 = False

    def __call__(self, img):
        if self._refImg is not None:
            print(self._refImg, self._pushed1, self._pushed2)
            self._refImg = cv2.imread(self._refImg)
            img = img.astype(np.float32) / 255.0
            self._refImg = self._refImg.astype(np.float32) / 255.0
            if img.shape != self._refImg.shape:
                self._refImg = skimage.transform.resize(self._refImg, img.shape, anti_aliasing=True)
            result = skimage.exposure.match_histograms(img, self._refImg)
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            return result
        else:
            return img


class NoiseItem(MyItem):
    def __init__(self, parent=None):
        super(NoiseItem, self).__init__('噪声生成', parent=parent)
        self._ksize = 3
        self._noise = GAUSSIAN_NOISE
        self._mean = 0
        self._variance = 500
        self._saltProb = 0.1
        self._pepperProb = 0.1

    def __call__(self, img):
        if self._noise == GAUSSIAN_NOISE:
            sigma = self._variance ** 0.5
            gaussianNoise = np.random.normal(self._mean, sigma, img.shape)
            noise_img = np.clip(img + gaussianNoise, 0, 255).astype(np.uint8)
            img = noise_img
        elif self._noise == SALT_AND_PEPPER:
            noisy = np.copy(img)
            totalPixels = img.size
            num_salt = np.ceil(self._saltProb * totalPixels)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
            noisy[coords[0], coords[1]] = 1
            num_pepper = np.ceil(self._pepperProb * totalPixels)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
            noisy[coords[0], coords[1]] = 0
            img = noisy
        elif self._noise == MOTION_BLUR:
            kernel = np.zeros((self._ksize, self._ksize))
            kernel[int((self._ksize - 1) / 2), :] = np.ones(self._ksize)
            kernel /= self._ksize
            motioned = cv2.filter2D(img, -1, kernel)
            img = motioned
        return img


class RestoreItem(MyItem):
    def __init__(self, parent=None):
        super(RestoreItem, self).__init__('修复图像', parent=parent)
        self._ksize = 1
        self._choice = GAUSSIAN_NOISE

    def __call__(self, img):
        if self._choice == ARITHMETRIC_MEAN:
            kSize = (self._ksize, self._ksize)
            kernalMean = np.ones(kSize, np.float32) / (kSize[0] * kSize[1])
            imgAriMean = cv2.filter2D(img, -1, kernalMean)
            img = imgAriMean
        elif self._choice == GEOMETRICMEAN:
            image = img.astype(np.float32)
            rows, cols, channels = image.shape
            output = np.zeros_like(image)
            pad_size = self._ksize // 2
            padded_image = np.pad(image, pad_width=((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
            for c in range(channels):
                for i in range(rows):
                    for j in range(cols):
                        neighborhood = padded_image[i:i + self._ksize, j:j + self._ksize, c]
                        positive_values = neighborhood[neighborhood > 0]
                        if positive_values.size > 0:
                            geometric_mean = np.exp(np.sum(np.log(positive_values)) / positive_values.size)
                        else:
                            geometric_mean = 0
                        output[i, j, c] = geometric_mean
            output = np.clip(output, 0, 255)
            img = output.astype(np.uint8)
        elif self._choice == ADAPTIVE_LOCAL:
            adpative_local = cv2.fastNlMeansDenoising(img, None, 30, 7, 21)
            img = adpative_local
        elif self._choice == MEDIAN_RESTORE:
            img = cv2.medianBlur(img, self._ksize)
        elif self._choice == MODIFIED_ALPHA:
            image = img.astype(np.float32)
            pad_size = self._ksize // 2
            padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
            output_image = np.zeros_like(image)
            kernel = np.ones((self._ksize, self._ksize)) / (self._ksize * self._ksize)
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    alpha = 0.5
                    window = padded_image[i:i + self._ksize, j:j + self._ksize]
                    local_value = image[i, j]
                    weights = 1 / (1 + alpha * np.abs(window - local_value))
                    weighted_sum = np.sum(weights * window, axis=(0, 1))
                    normalization_factor = np.sum(weights)
                    output_image[i, j] = weighted_sum / normalization_factor

            img = np.clip(output_image, 0, 255).astype(np.uint8)
        elif self._choice == ADAPTIVE_MEDIAN:
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError("Input image must be a color image with 3 channels.")
            hImg, wImg, channels = img.shape
            smax = 5
            m, n = smax, smax
            hPad = int((m - 1) / 2)
            wPad = int((n - 1) / 2)
            imgPad = np.pad(img.copy(), ((hPad, m - hPad - 1), (wPad, n - wPad - 1), (0, 0)), mode="edge")
            imgAdaMedFilter = np.zeros_like(img)
            for c in range(channels):
                for i in range(hPad, hPad + hImg):
                    for j in range(wPad, wPad + wImg):
                        ksize = 5
                        k = int(ksize / 2)
                        pad = imgPad[i - k:i + k + 1, j - k:j + k + 1, c]
                        zxy = img[i - hPad, j - wPad, c]
                        zmin = np.min(pad)
                        zmed = np.median(pad)
                        zmax = np.max(pad)
                        if zmin < zmed < zmax:
                            if zmin < zxy < zmax:
                                imgAdaMedFilter[i - hPad, j - wPad, c] = zxy
                            else:
                                imgAdaMedFilter[i - hPad, j - wPad, c] = zmed
                        else:
                            while True:
                                ksize += 2
                                if zmin < zmed < zmax or ksize > smax:
                                    break
                                k = int(ksize / 2)
                                pad = imgPad[i - k:i + k + 1, j - k:j + k + 1, c]
                                zmed = np.median(pad)
                                zmin = np.min(pad)
                                zmax = np.max(pad)
                            if zmin < zmed < zmax or ksize > smax:
                                if zmin < zxy < zmax:
                                    imgAdaMedFilter[i - hPad, j - wPad, c] = zxy
                                else:
                                    imgAdaMedFilter[i - hPad, j - wPad, c] = zmed
            img = imgAdaMedFilter.astype(img.dtype)
        elif self._choice == DECONVOLUTION:
            num_iter = 30
            kernel = np.ones((self._ksize, self._ksize)) / (self._ksize * self._ksize)
            img = img.astype(np.float32) / 255.0
            restored_img = np.zeros_like(img)
            for i in range(img.shape[2]):
                channel = img[:, :, i]
                restored_channel = restoration.richardson_lucy(channel, kernel, num_iter=num_iter)
                restored_img[:, :, i] = restored_channel
            restored_img = np.clip(restored_img * 255.0, 0, 255).astype(np.uint8)
            img = restored_img
        elif self._choice == WIENER_FILTER:
            img = img.astype(np.float32) / 255.0
            kernel = np.ones((self._ksize, self._ksize)) / (self._ksize * self._ksize)
            restored_img = np.zeros_like(img)
            for i in range(img.shape[2]):
                channel = img[:, :, i]
                restored_channel = restoration.wiener(channel, kernel, balance=0.1)
                restored_img[:, :, i] = restored_channel
            restored_img = np.clip(restored_img, 0, 1)
            restored_img = (restored_img * 255).astype(np.uint8)
            img = restored_img
        return img

class OperationItem(MyItem):
    def __init__(self, parent=None):
        super(OperationItem, self).__init__('图像运算', parent=parent)
        self._refImg = None
        self._pushed1 = False
        self._pushed2 = False
        self._choice = ADD

    def __call__(self, img):
        if self._refImg is not None:
            print(self._refImg, self._pushed1, self._pushed2)
            self._refImg = cv2.imread(self._refImg)
            img = img.astype(np.float32) / 255.0
            self._refImg = self._refImg.astype(np.float32) / 255.0
            self._refImg = skimage.transform.resize(self._refImg, img.shape, anti_aliasing=True)
            if self._choice == ADD:
                dst = cv2.add(src1=img, src2=self._refImg)
            elif self._choice == SUBTRACT:
                dst = cv2.subtract(src1=img, src2=self._refImg)
            elif self._choice == MULTIPLE:
                dst = cv2.multiply(src1=img, src2=self._refImg)
            elif self._choice == DIVIDE:
                dst = cv2.divide(src1=img, src2=self._refImg)
            result = np.clip(dst * 255, 0, 255).astype(np.uint8)
            return result
        else:
            return img

