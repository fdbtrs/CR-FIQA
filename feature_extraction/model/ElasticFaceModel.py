import os

from backbones.iresnet import iresnet100
from model.FaceModel import FaceModel
import torch
class ElasticFaceModel(FaceModel):
    def __init__(self, model_prefix, model_epoch, gpu_id):
        super(ElasticFaceModel, self).__init__(model_prefix, model_epoch, gpu_id)

    def _get_model(self, ctx, image_size, prefix, epoch, layer):
        weight = torch.load(os.path.join(prefix,epoch+"backbone.pth"))
        backbone = iresnet100().to(f"cuda:{ctx}")
        backbone.load_state_dict(weight)
        model = torch.nn.DataParallel(backbone, device_ids=[ctx])
        model.eval()
        return model

    @torch.no_grad()
    def _getFeatureBlob(self,input_blob):
        imgs = torch.Tensor(input_blob).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat = self.model(imgs)
        #feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()
