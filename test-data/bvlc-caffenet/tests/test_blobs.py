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
            raise FileNotFoundError('{} or {} not found'.format(self.deploy, self.caffemodel))

        self.mean = os.path.join(os.getcwd(), 'ilsvrc12/ilsvrc_2012_mean.npy')
        self.cat = os.path.join(os.getcwd(), 'images/cat.jpg')
        if not os.path.isfile(self.mean) or not os.path.isfile(self.cat):
            raise FileNotFoundError('{} or {} not found'.format(self.mean, self.cat))

        words = os.path.join(os.getcwd(), 'ilsvrc12/synset_words.txt')
        if not os.path.isfile(words):
            raise FileNotFoundError('{} not found'.format(word))
        self.labels = np.loadtxt(words, str, delimiter='\t')

        caffe.set_mode_cpu()
        net = caffe.Net(self.deploy, self.caffemodel, caffe.TEST)

        # load input and configure preprocessing
        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        transformer.set_mean('data', np.load(self.mean).mean(1).mean(1))
        transformer.set_transpose('data', (2,0,1))
        transformer.set_channel_swap('data', (2,1,0))
        transformer.set_raw_scale('data', 255.0)

        # batch is 1
        net.blobs['data'].reshape(1,3,227,227)

        #load the image in the data layer
        im = caffe.io.load_image(self.cat)
        net.blobs['data'].data[...] = transformer.preprocess('data', im)

        # forward
        out = net.forward()

        self.net = net
        self.out = out

    def test_cat_output(self):
        argmax = self.out['prob'].argmax()
        self.assertEqual(argmax, 281)
        self.assertEqual(self.labels[argmax], 'n02123045 tabby, tabby cat')

    def test_blob_conv2(self):
        self.assertIn('conv2', self.net.blobs)
        conv2 = self.net.blobs['conv2']
        self.assertEqual(conv2.data.shape, (1, 256, 27, 27))
        self.assertAlmostEqual(conv2.data[0, 0, 0, 7], 37.137928, places=6)
        self.assertAlmostEqual(conv2.data[0, 1, 3, 5], 24.611811, places=6)

    def test_transpose_padded_blob_conv2(self):
        base = 8
        blob = self.net.blobs['conv2']
        ndarr = blob.data
        npad = ((0, 0),) + tuple(map(lambda d: (0, base - (d % base)) if d % base > 0 else (0, 0), ndarr.shape[1:]))
        pndarr = np.pad(ndarr, pad_width=npad, mode='constant', constant_values=0)
        tpndarr = pndarr.transpose(0, 2, 3, 1)
        self.assertEqual(ndarr.shape, (1, 256, 27, 27))
        self.assertEqual(pndarr.shape, (1, 256, 32, 32))
        self.assertEqual(tpndarr.shape, (1, 32, 32, 256))
        self.assertEqual(ndarr.strides, (256 * 27 * 27 * 4, 27 * 27 * 4, 27 * 4, 4))
        self.assertEqual(pndarr.strides, (256 * 32 * 32 * 4, 32 * 32 * 4, 32 * 4, 4))
        self.assertEqual(tpndarr.strides, (256 * 32 * 32 * 4, 32 * 4, 4, 32 * 32 * 4))

        self.assertAlmostEqual(pndarr[0, 0, 0, 7], 37.137928, places=6)
        self.assertAlmostEqual(pndarr[0, 1, 3, 5], 24.611811, places=6)
        self.assertAlmostEqual(tpndarr[0, 0, 7, 0], 37.137928, places=6)
        self.assertAlmostEqual(tpndarr[0, 3, 5, 1], 24.611811, places=6)

    def test_ndarray_to_proto(self):
        base = 8
        blob = self.net.blobs['conv2']
        ndarr = blob.data
        npad = ((0, 0),) + tuple(map(lambda d: (0, base - (d % base)) if d % base > 0 else (0, 0), ndarr.shape[1:]))
        pndarr = np.pad(ndarr, pad_width=npad, mode='constant', constant_values=0)
        tpndarr = pndarr.transpose(0, 2, 3, 1)

        protob = caffe.io.array_to_blobproto(tpndarr)
        self.assertAlmostEqual(protob.data[(1) + (5) * 256 + (3) * 256 * 32], 24.611811, places=6)
        self.assertAlmostEqual(protob.data[(0) + (7) * 256 + (0) * 256 * 32], 37.137928, places=6)


if __name__ == "__main__":
    unittest.main()
