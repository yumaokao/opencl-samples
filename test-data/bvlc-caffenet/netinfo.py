# vim:fileencoding=UTF-8:ts=4:sw=4:sta:et:sts=4:ai
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

import numpy as np
import pydotplus as pydot


# Internal layer and blob styles.
LAYER_STYLE_DEFAULT = {'shape': 'record',
                       'fillcolor': '#6495ED',
                       'style': 'filled'}
NEURON_LAYER_STYLE = {'shape': 'record',
                      'fillcolor': '#90EE90',
                      'style': 'filled'}
BLOB_STYLE = {'shape': 'octagon',
              'fillcolor': '#E0E0E0',
              'style': 'filled'}

CAFFE_BASE = '/home/yumaokao/hubs/caffe'
DEPLOY_PROTOTXT_PATH = CAFFE_BASE + '/models/bvlc_reference_caffenet/deploy.prototxt'


def choose_color_by_layertype(layertype):
    """Define colors for nodes based on the layer type.
    """
    color = '#6495ED'  # Default
    if layertype == 'Convolution' or layertype == 'Deconvolution':
        color = '#FF5050'
    elif layertype == 'Pooling':
        color = '#FF9900'
    elif layertype == 'InnerProduct':
        color = '#CC33FF'
    return color


def get_edge_label(layer):
    """Define edge label based on layer type.
    """
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
    """Get dictionary mapping pooling type number to type name
    """
    desc = caffe_pb2.PoolingParameter.PoolMethod.DESCRIPTOR
    d = {}
    for k, v in desc.values_by_name.items():
        d[v.number] = k
    return d


def get_layer_label(layer, rankdir):
    """Define node label based on layer type.

    Parameters
    ----------
    layer : ?
    rankdir : {'LR', 'TB', 'BT'}
        Direction of graph layout.

    Returns
    -------
    string :
        A label for the current layer
    """

    if rankdir in ('TB', 'BT'):
        # If graph orientation is vertical, horizontal space is free and
        # vertical space is not; separate words with spaces
        separator = ' '
    else:
        # If graph orientation is horizontal, vertical space is free and
        # horizontal space is not; separate words with newlines
        separator = '\\n'

    if layer.type == 'Convolution' or layer.type == 'Deconvolution':
        # Outer double quotes needed or else colon characters don't parse
        # properly
        node_label = '"%s%s(%s)%skernel size: %d%sstride: %d%spad: %d"' %\
                     (layer.name,
                      separator,
                      layer.type,
                      separator,
                      layer.convolution_param.kernel_size[0] if len(layer.convolution_param.kernel_size) else 1,
                      separator,
                      layer.convolution_param.stride[0] if len(layer.convolution_param.stride) else 1,
                      separator,
                      layer.convolution_param.pad[0] if len(layer.convolution_param.pad) else 0)
    elif layer.type == 'Pooling':
        pooling_types_dict = get_pooling_types_dict()
        node_label = '"%s%s(%s %s)%skernel size: %d%sstride: %d%spad: %d"' %\
                     (layer.name,
                      separator,
                      pooling_types_dict[layer.pooling_param.pool],
                      layer.type,
                      separator,
                      layer.pooling_param.kernel_size,
                      separator,
                      layer.pooling_param.stride,
                      separator,
                      layer.pooling_param.pad)
    else:
        node_label = '"%s%s(%s)"' % (layer.name, separator, layer.type)
    return node_label


def get_pydot_graph(caffe_net, rankdir, label_edges=True, phase=None):
    pydot_graph = pydot.Dot(caffe_net.name if caffe_net.name else 'Net',
                            graph_type='digraph',
                            rankdir=rankdir)
    pydot_nodes = {}
    pydot_edges = []
    for layer in caffe_net.layer:
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
            pydot_nodes[bottom_blob + '_blob'] = pydot.Node('%s' % bottom_blob, **BLOB_STYLE)
            edge_label = '""'
            pydot_edges.append({'src': bottom_blob + '_blob',
                                'dst': node_name,
                                'label': edge_label})
        for top_blob in layer.top:
            pydot_nodes[top_blob + '_blob'] = pydot.Node('%s' % (top_blob))
            if label_edges:
                edge_label = get_edge_label(layer)
            else:
                edge_label = '""'
            pydot_edges.append({'src': node_name,
                                'dst': top_blob + '_blob',
                                'label': edge_label})
    # Now, add the nodes and edges to the graph.
    for node in pydot_nodes.values():
        pydot_graph.add_node(node)
    for edge in pydot_edges:
        pydot_graph.add_edge(
            pydot.Edge(pydot_nodes[edge['src']], pydot_nodes[edge['dst']], label=edge['label']))
    return pydot_graph


def main():
    netpara = caffe_pb2.NetParameter()
    text_format.Merge(open(DEPLOY_PROTOTXT_PATH).read(), netpara)
    net = caffe.Net(DEPLOY_PROTOTXT_PATH, caffe.TEST)

    graph = get_pydot_graph(netpara, 'LR', phase=caffe_pb2.Phase.Value('TEST'))
    with open('result.png', 'wb') as fp:
        fp.write(graph.create(format='png'))
    # import ipdb
    # ipdb.set_trace()


if __name__ == "__main__":
    main()
