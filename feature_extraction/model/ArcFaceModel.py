import mxnet as mx

from model.FaceModel import FaceModel


class ArcFaceModel(FaceModel):
    def __init__(self, model_prefix, model_epoch, gpu_id):
        super(ArcFaceModel, self).__init__(model_prefix, model_epoch, gpu_id)
    def _get_model(self, ctx, image_size, prefix, epoch, layer):
        print('loading', prefix, epoch)
        if ctx>=0:
            ctx = mx.gpu(ctx)
        else:
            ctx = mx.cpu()
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, int(epoch))
        all_layers = sym.get_internals()
        sym = all_layers[layer + '_output']
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
        model.set_params(arg_params, aux_params)
        return model

    def _getFeatureBlob(self,input_blob):
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.model.forward(db, is_train=False)
        emb = self.model.get_outputs()[0].asnumpy()
        return  emb