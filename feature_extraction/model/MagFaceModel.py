import os
from backbones.mag_network_inf import builder_inf
from model.FaceModel import FaceModel
import torch

class MagFaceModel(FaceModel):
    def __init__(self, model_prefix, model_epoch, gpu_id):
        super(MagFaceModel, self).__init__(model_prefix, model_epoch, gpu_id)

    def _get_model(self, ctx, image_size, prefix, epoch, layer):
        backbone = builder_inf(os.path.join(prefix,"magface_epoch_"+epoch+".pth"), "iresnet100", 512)
        model = torch.nn.DataParallel(backbone, device_ids=[ctx])
        model.eval()
        return model

    @torch.no_grad()
    def _getFeatureBlob(self,input_blob):
        imgs = torch.Tensor(input_blob).cuda()
        imgs.div_(255)
        feat = self.model(imgs)
        #feat = feat.reshape([self.batch_size, 2 * feat.shape[1]])
        return feat.cpu().numpy()
