# vim:fileencoding=UTF-8:ts=4:sw=4:sta:et:sts=4:ai
import unittest
import os
import caffe
import numpy as np


class TestCaffeNetBlob(unittest.TestCase):
    def setUp(self):
        # load bvlc_reference_caffenet
        print(os.getcwd())
        caffenet_path = os.path.join(os.getcwd(), 'models/bvlc_reference_caffenet')
        self.deploy = os.path.join(caffenet_path, 'deploy.prototxt')
        self.caffemodel = os.path.join(caffenet_path, 'bvlc_reference_caffenet.caffemodel')
        if not os.path.isfile(self.deploy) or not os.path.isfile(self.caffemodel):
            raise FileNotFoundError('{} or {} not found in {}'.format(self.deploy, self.caffemodel, caffenet_path))

        caffe.set_mode_cpu()
        net = caffe.Net(self.deploy, self.caffemodel, caffe.TEST)

        import ipdb
        ipdb.set_trace()
        pass

    def test_reshape_net(self):
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
