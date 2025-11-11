import torch
import torch.nn as nn
import torch.nn.functional as F


class InterLoss(nn.Module):
    def __init__(self, temp=0.01):
        super(InterLoss, self).__init__()
        self.temp = temp
        
    def forward(self, mul_emb, eng_emb, text_emb, vision_emb):
        # mul_emb: (batch_size, 512)
        # eng_emb: (batch_size, 512)
        # text_emb: (batch_size, 512)
        # vision_emb: (batch_size, 512)
        batch_size = mul_emb.size(0)
        
        # similarity matrix
        sim_eng_text = torch.matmul(eng_emb, text_emb.T) / self.temp
        sim_text_eng = torch.matmul(text_emb, eng_emb.T) / self.temp
        
        sim_mul_vision = torch.matmul(mul_emb, vision_emb.T) / self.temp
        sim_vision_kor = torch.matmul(vision_emb, mul_emb.T) / self.temp

        labels = torch.arange(batch_size).to(mul_emb.device)

        loss_eng_text = (F.cross_entropy(sim_eng_text, labels) + F.cross_entropy(sim_text_eng, labels)) / 2
        loss_mul_vision = (F.cross_entropy(sim_mul_vision, labels) + F.cross_entropy(sim_vision_kor, labels)) / 2
        return loss_eng_text, loss_mul_vision


class IntraLoss(nn.Module):
    def __init__(self):
        super(IntraLoss, self).__init__()
        
    def forward(self, mul_emb, eng_emb, text_emb, vision_emb):
        # mul_emb: (batch_size, 512)
        # eng_emb: (batch_size, 512)
        # text_emb: (batch_size, 512)
        # vision_emb: (batch_size, 512)
        batch_size = mul_emb.size(0)
        
        dist_mul_eng = F.pairwise_distance(mul_emb, eng_emb, p=2) 
        dist_text_vision = F.pairwise_distance(text_emb, vision_emb, p=2)

        loss = (dist_mul_eng + dist_text_vision).mean() / 2
        return loss


if __name__ == "__main__":
    # Example usage
    mul_emb = F.normalize(torch.randn(32, 512), dim=-1)
    eng_emb = F.normalize(torch.randn(32, 512), dim=-1)
    text_emb = F.normalize(torch.randn(32, 512), dim=-1)
    vision_emb = F.normalize(torch.randn(32, 512), dim=-1)

    inter_loss_fn = InterLoss(temp=0.1)
    intra_loss_fn = IntraLoss()

    inter_loss = inter_loss_fn(mul_emb, eng_emb, text_emb, vision_emb)
    intra_loss = intra_loss_fn(mul_emb, eng_emb, text_emb, vision_emb)

    print("Inter Loss:", inter_loss.item())
    print("Intra Loss:", intra_loss.item())

    