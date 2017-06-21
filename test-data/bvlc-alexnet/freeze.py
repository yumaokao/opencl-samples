# vim:fileencoding=UTF-8:ts=4:sw=4:sta:et:sts=4:ai
import caffe
import numpy as np
import colorama

CAFFE_BASE = '/home/yumaokao/hubs/caffe'
DEPLOY_PROTOTXT_PATH = CAFFE_BASE + '/models/bvlc_reference_caffenet/deploy.prototxt'
CAFFEMODEL_PATH = CAFFE_BASE + '/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'


def pretty(name, data, channel=0, stride=2, color=False):
    print()
    print('{}: [{}] - [1, {}, :, :]'.format(name, data.shape, channel))
    for i, v in np.ndenumerate(data[0, channel, ...]):
        if color and (i[1] + 1) % stride == 0 and (i[0] + 1) % stride == 0:
            print(colorama.Fore.CYAN, end='')
        print('{:5.2f}'.format(v), end=' ')
        if color:
            print(colorama.Style.RESET_ALL, end='')
        if (i[1] + 1) % data.shape[2] == 0:
            print()

def convpool(net, conv, pool, chan=None):
    channels = net.blobs[conv].channels
    if channels != net.blobs[pool].channels:
        raise ValueError('conv and pool layers should have same channels')
    colorama.init()
    if chan is None:
        for c in range(channels):
            pretty(conv, net.blobs[conv].data, channel=c, stride=2, color=True)
            pretty(pool, net.blobs[pool].data, channel=c)
    else:
        c = chan if chan < channels else 0
        pretty(conv, net.blobs[conv].data, channel=c, stride=2, color=True)
        pretty(pool, net.blobs[pool].data, channel=c)


def main():
    net = caffe.Net(DEPLOY_PROTOTXT_PATH, CAFFEMODEL_PATH, caffe.TEST)

    # load input and configure preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load(CAFFE_BASE + '/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)

    # batch is 1
    net.blobs['data'].reshape(1,3,227,227)

    #load the image in the data layer
    im = caffe.io.load_image(CAFFE_BASE + '/examples/images/cat.jpg')
    net.blobs['data'].data[...] = transformer.preprocess('data', im)

    # forward
    out = net.forward()

    # ## result verification
    '''
    # predicted predicted class
    print(out['prob'].argmax())

    # print predicted labels
    labels = np.loadtxt(CAFFE_BASE + '/data/ilsvrc12/synset_words.txt', str, delimiter='\t')
    top_k = net.blobs['prob'].data[0].flatten().argsort()[-1:-6:-1]
    print(labels[top_k])
    '''

    # ## pretty print conv2, pool2
    convpool(net, 'conv2', 'pool2', chan=1)

    # import ipdb
    # ipdb.set_trace()


if __name__ == "__main__":
    main()
