import cv2
import numpy as np

from skimage import feature
from skimage.filters import threshold_sauvola, threshold_otsu, apply_hysteresis_threshold


class Pipeline:
    def __init__(self, data):
        self.data = data

    def _resize(self):
        self.data = cv2.resize(self.data, dsize=(91, 124), interpolation=cv2.INTER_CUBIC)

    def _luminance(self):
        # 0.2126*R + 0.7152*G + 0.0722*B
        self.data = np.dot(self.data, [0.0722, 0.7152, 0.2126]).astype(np.uint8)


class LocalBinaryPatternPipeline(Pipeline):
    def __init__(self, data, p, r, method):
        super().__init__(data)
        self.p = p
        self.r = r
        self.method = method

    def transform(self):
        self._luminance()
        self._lbp()
        self._resize()
        return self.data

    def _lbp(self):
        data = feature.local_binary_pattern(self.data, self.p, self.r, method=self.method)
        self.data = data


class SauvolaPipeline(Pipeline):
    def __init__(self, data):
        super().__init__(data)

    def transform(self):
        self._luminance()
        self._sauvola()
        self._resize()
        return self.data

    def _sauvola(self):
        self.data = self.data > threshold_sauvola(self.data)


class OtsuPipeline(Pipeline):
    def __init__(self, data):
        super().__init__(data)

    def transform(self):
        self._luminance()
        self._otsu()
        self._resize()
        return self.data

    def _otsu(self):
        self.data = self.data > threshold_otsu(self.data)


class CannyPipeline(Pipeline):
    def __init__(self, data, low, high):
        super().__init__(data)
        self.low = low
        self.high = high

    def transform(self):
        self._luminance()
        self._canny()
        self._resize()
        return self.data

    def _canny(self):
        self.data = apply_hysteresis_threshold(self.data, self.low, self.high)


class AdaptiveGaussianPipeline(Pipeline):
    def __init__(self, data):
        super().__init__(data)

    def transform(self):
        self._luminance()
        self._adaptive_gaussian()
        self._resize()
        return self.data

    def _adaptive_gaussian(self):
        self.data = cv2.adaptiveThreshold(self.data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
