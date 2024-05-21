import torch
import torch.nn as nn
import torchvision.models as models
from modules.FLAIR import FLAIRModel


class VisualExtractor(nn.Module):
    def __init__(self, args):
        super(VisualExtractor, self).__init__()
        if args.FLAIR:
            model = FLAIRModel(from_checkpoint=True, weights_path=args.FLAIR_path,
                      projection=2048, norm_features=True,
                      vision_pretrained=False)
            vision_model = model.vision_model.model
            vision_model = torch.nn.Sequential(*(list(vision_model.children())[:-2]))
            self.model = vision_model
        else:
            self.visual_extractor = args.visual_extractor
            self.pretrained = args.visual_extractor_pretrained
            model = getattr(models, self.visual_extractor)(pretrained=self.pretrained)
            modules = list(model.children())[:-2]
            self.model = nn.Sequential(*modules).to('cuda')
        self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

    def forward(self, images):
        patch_feats = self.model(images)
        avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1))
        batch_size, feat_size, _, _ = patch_feats.shape
        patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        return patch_feats, avg_feats
