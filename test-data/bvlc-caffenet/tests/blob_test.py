# vim:fileencoding=UTF-8:ts=4:sw=4:sta:et:sts=4:ai
import unittest
import caffe
import numpy as np
import colorama

CAFFE_BASE = '/home/yumaokao/hubs/caffe'
DEPLOY_PROTOTXT_PATH = CAFFE_BASE + '/models/bvlc_reference_caffenet/deploy.prototxt'
CAFFEMODEL_PATH = CAFFE_BASE + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'


class TestCaffeNetBlob(unittest.TestCase):
    def SetUp(self):
        # download bvlc_reference_caffenet 


if __name__ == "__main__":
    unittest.main()
