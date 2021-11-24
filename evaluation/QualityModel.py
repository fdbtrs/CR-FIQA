import os

from backbones.iresnet import iresnet100, iresnet50
from evaluation.FaceModel import FaceModel
import torch
class QualityModel(FaceModel):
    def __init__(self, model_prefix, model_epoch, gpu_id):
        super(QualityModel, self).__init__(model_prefix, model_epoch, gpu_id)

    def _get_model(self, ctx, image_size, prefix, epoch, layer, backbone):
        weight = torch.load(os.path.join(prefix,epoch+"backbone.pth"))
        if (backbone=="iresnet50"):
            backbone = iresnet50(num_features=512, qs=1, use_se=False).to(f"cuda:{ctx}")
        else:
            backbone = iresnet100(num_features=512, qs=1, use_se=False).to(f"cuda:{ctx}")

        backbone.load_state_dict(weight)
        model = torch.nn.DataParallel(backbone, device_ids=[ctx])
        model.eval()
        return model

    @torch.no_grad()
    def _getFeatureBlob(self,input_blob):
        imgs = torch.Tensor(input_blob).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat, qs = self.model(imgs)
        return feat.cpu().numpy(), qs.cpu().numpy()
