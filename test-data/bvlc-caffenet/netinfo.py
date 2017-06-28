# vim:fileencoding=UTF-8:ts=4:sw=4:sta:et:sts=4:ai
import caffe
import caffe.draw
from caffe.proto import caffe_pb2
from google.protobuf import text_format

import numpy as np
import colorama


CAFFE_BASE = '/home/yumaokao/hubs/caffe'
DEPLOY_PROTOTXT_PATH = CAFFE_BASE + '/models/bvlc_reference_caffenet/deploy.prototxt'


def main():
    netpara = caffe_pb2.NetParameter()
    text_format.Merge(open(DEPLOY_PROTOTXT_PATH).read(), netpara)
    # graph = caffe.draw.draw_net(netpara, 'LR', ext='dot', phase=caffe_pb2.Phase.Value('TEST'))
    graph = caffe.draw.get_pydot_graph(netpara, 'LR', phase=caffe_pb2.Phase.Value('TEST'))
    print(graph)
    import ipdb
    ipdb.set_trace()

    net = caffe.Net(DEPLOY_PROTOTXT_PATH, caffe.TEST)
    print(net.blobs['conv2'].data.shape)


if __name__ == "__main__":
    main()
