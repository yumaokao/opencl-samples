# vim:fileencoding=UTF-8:ts=4:sw=4:sta:et:sts=4:ai
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

import argparse
import numpy as np
import pydotplus as pydot


# Internal layer and blob styles.
LAYER_STYLE_DEFAULT = {'shape': 'record', 'fillcolor': '#6495ED', 'style': 'filled'}
NEURON_LAYER_STYLE = {'shape': 'record', 'fillcolor': '#90EE90', 'style': 'filled'}
BLOB_STYLE = {'shape': 'octagon', 'fillcolor': '#E0E0E0', 'style': 'filled'}

VERSION = '0.3.0'
CAFFE_BASE = '/home/yumaokao/hubs/caffe'
DEPLOY_PROTOTXT_PATH = CAFFE_BASE + '/models/bvlc_reference_caffenet/deploy.prototxt'


def choose_color_by_layertype(layertype):
    color = '#6495ED'  # Default
    if layertype == 'Convolution' or layertype == 'Deconvolution':
        color = '#FF5050'
    elif layertype == 'Pooling':
        color = '#FF9900'
    elif layertype == 'InnerProduct':
        color = '#CC33FF'
    return color


def get_edge_label(layer):
    if layer.type == 'Data':
        edge_label = 'Batch ' + str(layer.data_param.batch_size)
    elif layer.type == 'Convolution' or layer.type == 'Deconvolution':
        edge_label = str(layer.convolution_param.num_output)
    elif layer.type == 'InnerProduct':
        edge_label = str(layer.inner_product_param.num_output)
    else:
        edge_label = '""'

    return edge_label


def get_pooling_types_dict():
    desc = caffe_pb2.PoolingParameter.PoolMethod.DESCRIPTOR
    d = {}
    for k, v in desc.values_by_name.items():
        d[v.number] = k
    return d


def get_layer_label(layer, rankdir):
    separator = ' ' if rankdir in ('TB', 'BT') else '\\n'

    if layer.type == 'Convolution' or layer.type == 'Deconvolution':
        param = layer.convolution_param
        label = '"{name}{s}[{type}]{s}{kernel}x{kernel}_{stride}(S){s}pad {pad}"'
        node_label = label.format(name=layer.name, type=layer.type, s=separator,
                                  kernel=param.kernel_size[0] if len(param.kernel_size) else 1,
                                  stride=param.stride[0] if len(param.stride) else 1,
                                  pad=param.pad[0] if len(param.pad) else 0)
    elif layer.type == 'Pooling':
        pooling_types_dict = get_pooling_types_dict()
        param = layer.pooling_param
        label = '"{name}{s}[{type}]{s}{pool}{s}{kernel}x{kernel}_{stride}(S){s}pad {pad}"'
        node_label = label.format(name=layer.name, type=layer.type, s=separator,
                                  pool=pooling_types_dict[layer.pooling_param.pool],
                                  kernel=param.kernel_size,
                                  stride=param.stride,
                                  pad=param.pad)
    else:
        label = '"{name}{s}[{type}]"'
        node_label = label.format(name=layer.name, type=layer.type, s=separator)
    return node_label


def get_blob_label(blobname, net, rankdir):
    separator = ' ' if rankdir in ('TB', 'BT') else '\\n'

    if blobname not in net.blobs:
        raise ValueError('Blob ' + blobname + ' not in caffe net')

    blob = net.blobs[blobname]

    label = '"{name}{s}{shape}{s}= {size}{s}= {sizeM:.2f} M"'
    dims = [str(s) for s in blob.data.shape]
    shape = ','.join(dims)
    blob_label = label.format(name=blobname, shape=shape, s=separator,
                              size=blob.data.size, sizeM=blob.data.size / (1024 * 1024))
    return blob_label


def get_pydot_graph(net, netpara, rankdir, label_edges=True, phase=None):
    pydot_graph = pydot.Dot(netpara.name if netpara.name else 'Net',
                            graph_type='digraph',
                            rankdir=rankdir)
    pydot_nodes = {}
    pydot_edges = []
    for layer in netpara.layer:
        if phase is not None:
            included = False
            if len(layer.include) == 0:
                included = True
            if len(layer.include) > 0 and len(layer.exclude) > 0:
                raise ValueError('layer ' + layer.name + ' has both include and exclude specified.')
            for layer_phase in layer.include:
                included = included or layer_phase.phase == phase
            for layer_phase in layer.exclude:
                included = included and not layer_phase.phase == phase
            if not included:
                continue

        node_label = get_layer_label(layer, rankdir)
        node_name = "%s_%s" % (layer.name, layer.type)
        if (len(layer.bottom) == 1 and len(layer.top) == 1 and layer.bottom[0] == layer.top[0]):
            # We have an in-place neuron layer.
            pydot_nodes[node_name] = pydot.Node(node_label, **NEURON_LAYER_STYLE)
        else:
            layer_style = LAYER_STYLE_DEFAULT
            layer_style['fillcolor'] = choose_color_by_layertype(layer.type)
            pydot_nodes[node_name] = pydot.Node(node_label, **layer_style)

        for bottom_blob in layer.bottom:
            pydot_nodes[bottom_blob + '_blob'] = pydot.Node(get_blob_label(bottom_blob, net, rankdir), **BLOB_STYLE)
            edge_label = '""'
            pydot_edges.append({'src': bottom_blob + '_blob', 'dst': node_name, 'label': edge_label})
        for top_blob in layer.top:
            pydot_nodes[top_blob + '_blob'] = pydot.Node(get_blob_label(top_blob, net, rankdir))
            edge_label = get_edge_label(layer) if label_edges else '""'
            pydot_edges.append({'src': node_name, 'dst': top_blob + '_blob', 'label': edge_label})

    # Now, add the nodes and edges to the graph.
    for node in pydot_nodes.values():
        pydot_graph.add_node(node)
    for edge in pydot_edges:
        pydot_graph.add_edge(
            pydot.Edge(pydot_nodes[edge['src']], pydot_nodes[edge['dst']], label=edge['label']))
    return pydot_graph


def main():
    parser = argparse.ArgumentParser(description='hamiorg')
    parser.add_argument('-v', '--verbose', help='show more debug information', action='count', default=0)
    parser.add_argument('-V', '--version', action='version', version=VERSION, help='show version infomation')
    parser.add_argument('-n', '--batch_size', default=8, help='batch size')
    parser.add_argument('-p', '--plot', action='store_true', help='plot the network')
    parser.add_argument('-P', '--pooling', action='store_true', help='calculate pooling layers')
    parser.add_argument('deploy', help='deploy.prototxt for the caffe model')
    args = parser.parse_args()

    netpara = caffe_pb2.NetParameter()
    text_format.Merge(open(args.deploy).read(), netpara)
    net = caffe.Net(args.deploy, caffe.TEST)

    # reshape for batch size
    net.blobs['data'].reshape(args.batch_size, 3, 227, 227)
    net.reshape()

    if args.plot:
        graph = get_pydot_graph(net, netpara, 'LR', phase=caffe_pb2.Phase.Value('TEST'))
        with open('result.png', 'wb') as fp:
            fp.write(graph.create(format='png'))

    if args.pooling:
        print('YMK')

    # import ipdb
    # ipdb.set_trace()


if __name__ == "__main__":
    main()
