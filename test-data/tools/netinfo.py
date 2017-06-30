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


def pooling_info(layer, net):
    # parameters
    clusters = 1
    cu_groups_per_cluster = 4
    gsm_size = 1 * 1024 * 1024
    bus_util = 0.8

    print(layer.name)

    param = layer.pooling_param
    kernel_size = param.kernel_size
    cycles_uint8_in_16b = 2 # TODO: 2 if param.pool == 'MAX' else 1
    cus = clusters * cu_groups_per_cluster

    pixels = sum(map(lambda b: net.blobs[b].data.size, layer.top))

    # ideal computation cycles
    ideal_cycles = pixels * kernel_size * kernel_size // (128 * cus)

    # estimated computation cycles needed
    alu_cycles = (pixels * kernel_size * kernel_size * cycles_uint8_in_16b // (128 * cus)) + 60

    # estimated load store cycles needed
    store_pixels = pixels
    load_pixels = pixels * kernel_size * kernel_size
    load_store_cycles = (load_pixels + store_pixels) * cycles_uint8_in_16b * 1.1 // (128 * cus)

    # dram_gsm_cycles
    src_pixels = sum(map(lambda b: net.blobs[b].data.size, layer.bottom))
    if src_pixels < 0.5 * gsm_size:
        dram_gsm_cycles = 0
    else:
        dram_gsm_cycles = (src_pixels + pixels) * 1.05 // (16 * clusters * bus_util)

    # estimated cycles
    estimated_cycles = max(load_store_cycles, alu_cycles, dram_gsm_cycles)

    # estimated_util
    estimated_util = ideal_cycles / estimated_cycles

    print('ideal cycles: {}'.format(ideal_cycles))
    print('  alu_cycles: {}'.format(alu_cycles))
    print('  load_store_cycles: {}'.format(load_store_cycles))
    print('  dram_gsm_cycles: {}'.format(dram_gsm_cycles))
    print('estimated_cycles: {}'.format(estimated_cycles))
    print('estimated_util: {:.2%}'.format(estimated_util))
    print()

    return ideal_cycles, estimated_cycles


class Utilization():
    def __init__(self, name='Layer'):
        self.name = name
        self.total_ideal = 0.0
        self.total_estimated = 0.0

    def add(self, ideal, estimated):
        self.total_ideal += ideal
        self.total_estimated += estimated

    def util(self):
        return self.total_ideal / self.total_estimated

    def info(self):
        print('Utilization: {}'.format(self.name))
        print('  ideal cycles: {}'.format(self.total_ideal))
        print('  estimated cycles: {}'.format(self.total_estimated))
        print('  estimated util: {:.2%}'.format(self.util()))


def main():
    parser = argparse.ArgumentParser(description='hamiorg')
    parser.add_argument('-v', '--verbose', help='show more debug information', action='count', default=0)
    parser.add_argument('-V', '--version', action='version', version=VERSION, help='show version infomation')
    parser.add_argument('-n', '--batch_size', type=int, default=1, help='batch size')
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
        print()
        print('Show Pooling Layer Infomation')
        util = Utilization(name='Pooling')
        for layer in netpara.layer:
            if layer.type == 'Pooling':
                util.add(*pooling_info(layer, net))
        util.info()


    # import ipdb
    # ipdb.set_trace()


if __name__ == "__main__":
    main()
