#!/bin/python
from google.protobuf import text_format
from caffe.proto import caffe_pb2 as pb

net = pb.NetParameter()
net_str = str(open('./models/clef_googlenet/c_to_i/train_val.prototxt', 'rb').read())

text_format.Parse(net_str, net)

# lr_mult and decay_mult for inception layers
inceptions = {
    'inception_3a': [[0, 0], [0, 0]],
    'inception_3b': [[0, 0], [0, 0]],
    'inception_4a': [[0, 0], [0, 0]],
    'inception_4b': [[0, 0], [0, 0]],
    'inception_4c': [[0, 0], [0, 0]],
    'inception_4d': [[0, 0], [0, 0]],
    'inception_4e': [[0, 0], [0, 0]],
    'inception_5a': [[0, 0], [0, 0]],
    'inception_5b': [[0, 0], [0, 0]],
}
# lr_mult and decay_mult for 3 loss layers
losses = {
    'loss1/conv': [[0, 0], [0, 0]],
    'loss1/fc': [[0, 0], [0, 0]],
    'loss1/classifier_office': [[0, 0], [0, 0]],
    'loss2/conv': [[0, 0], [0, 0]],
    'loss2/fc': [[0, 0], [0, 0]],
    'loss2/classifier_office': [[0, 0], [0, 0]],
    'loss3/classifier_office': [[1, 2], [1, 0]],
}
# entropy for entropy layer
entropy = {
    'entropy_loss': [0, 15000],
}
# mmd_lambda for 4 mmd layers
mmds = {
    'mmd3/4d': 0,
    'mmd3/4e': 0,
    'mmd3/5a': 0,
    'mmd3/loss3': 5,
}
# change lr_mult for inception layers
for layer in net.layer:
    name = layer.name
    name_prefic = layer.name.split('/')[0]

    if name_prefic in inceptions:
        params = inceptions[name_prefic]
        if len(layer.param) == 2:
            print layer.type
            layer.param[0].lr_mult = params[0][0]
            layer.param[1].lr_mult = params[0][1]
            layer.param[0].decay_mult = params[1][0]
            layer.param[1].decay_mult = params[1][1]

    if name in losses:
        params = losses[name]
        if len(layer.param) == 2:
            print 'loss+' + layer.type
            layer.param[0].lr_mult = params[0][0]
            layer.param[1].lr_mult = params[0][1]
            layer.param[0].decay_mult = params[1][0]
            layer.param[1].decay_mult = params[1][1]
        else:
            print 'error'

    if name in mmds:
        mmd_lambda = mmds[name]
        layer.mmd_param.mmd_lambda = mmd_lambda

    if name in entropy:
        layer.loss_weight[0] = entropy[name][0]
        layer.entropy_param.iterations_num = entropy[name][1]

# write to file
output = text_format.MessageToString(net)
out_file = open('./models/clef_googlenet/c_to_i/train_val.prototxt', 'w')
out_file.write(output)
out_file.close()

