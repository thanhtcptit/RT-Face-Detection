import cv2
import numpy as np
import mxnet as mx

from sklearn import preprocessing


def do_flip(data):
    for idx in range(data.shape[0]):
        data[idx, :, :] = np.fliplr(data[idx, :, :])


def get_model(ctx, image_size, prefix, epoch, layer):
    print('loading ', prefix)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)
    return model


class Face2VecModel:
    def __init__(self, prefix, epoch, image_size=[112, 112], gpu=-1):
        if gpu >= 0:
            ctx = mx.gpu(gpu)
        else:
            ctx = mx.cpu()
        image_size = image_size
        self.model = get_model(ctx, image_size, prefix, epoch, 'fc1')
        self.image_size = image_size

    def get_feature(self, aligned):
        if len(aligned.shape) == 3:
            input_blob = np.expand_dims(aligned, axis=0)
        else:
            input_blob = aligned
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        outputs = self.model.get_outputs()[0].asnumpy()
        embeddings = preprocessing.normalize(outputs)
        return embeddings
