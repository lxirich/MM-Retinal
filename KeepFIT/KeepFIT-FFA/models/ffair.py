import torch
import torch.nn as nn
import numpy as np

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder


class FFAIRModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(FFAIRModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets=None, mode='train'):
        att_feats = 0
        fc_feats = 0
        for ind in range(images.shape[1]):
            att_feats_new, fc_feats_new = self.visual_extractor(images[:, ind])
            att_feats += att_feats_new
            fc_feats += fc_feats_new
        att_feats /= images.shape[1]
        fc_feats /= images.shape[1]
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

