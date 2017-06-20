# vim:fileencoding=UTF-8:ts=4:sw=4:sta:et:sts=4:ai
import caffe

DEPLOY_PROTOTXT_PATH = '/home/yumaokao/hubs/caffe/models/bvlc_alexnet/deploy.prototxt'
CAFFEMODEL_PATH = '/home/yumaokao/hubs/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel'


def main():
    net = caffe.Net(DEPLOY_PROTOTXT_PATH, CAFFEMODEL_PATH, caffe.TEST)


if __name__ == "__main__":
    main()
