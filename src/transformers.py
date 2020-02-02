import cv2
import numpy as np

from skimage import feature


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
