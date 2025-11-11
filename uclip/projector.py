from torch import nn
import torch.nn.functional as F

class uCLIP_Head(nn.Module):
    def __init__(self, args):
        super().__init__()
        if args.model == 'minilm':
            self.Head_A = nn.Sequential(
                nn.Linear(384, 384 * 2),
                nn.BatchNorm1d(384*2),
                nn.ReLU(),
                nn.Linear(384*2, 512),
            )
        elif args.model == 'e5' or args.model == 'mpnet' or args.model == 'xlmr':
            self.Head_A = nn.Sequential(
                nn.Linear(768, 768*2),
                nn.BatchNorm1d(768*2),
                nn.ReLU(),
                nn.Linear(768*2, 512),
            )
            
        if args.vlm_model == 'clip' or args.vlm_model == 'openclip':
            self.Head_B = nn.Sequential(
                nn.Linear(512, 512 * 2),
                nn.BatchNorm1d(512*2),
                nn.ReLU(),
                nn.Linear(512*2, 512),
            )
        elif args.vlm_model == 'siglip2':
            self.Head_B = nn.Sequential(
                nn.Linear(768, 768 * 2),
                nn.BatchNorm1d(768*2),
                nn.ReLU(),
                nn.Linear(768*2, 512),
            )

    def forward_text(self, x):
        """
        Input: x - (batch_size, text_input_dim)
        Output: (batch_size, proj_dim), normalized
        """
        x = self.Head_A(x)
        return F.normalize(x, dim=-1)

    def forward_clip(self, x):
        """
        Input: x - (batch_size, clip_input_dim)
        Output: (batch_size, proj_dim), normalized
        """
        x = self.Head_B(x)
        return F.normalize(x, dim=-1)