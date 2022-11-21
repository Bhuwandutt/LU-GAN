Encoder.py
Details
Activity


import torch
from torch import nn
from transformers import AutoModel


class Encoder(nn.Module):

    def __init__(self, feature_base_dim):

        super(Encoder, self).__init__()

        self.feature_base_dim = feature_base_dim
        self.batch_first = True
        self.device = 'cuda'

        self.sentence_bert = None

        self.model = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-base').to(device=self.device)

        self.fc = nn.Sequential(
            # nn.Linear(2 * self.hidden_size * self.num_dir, self.feature_base_dim, bias=False),
            nn.Linear(1536, self.feature_base_dim),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self,finding_input_ids,impression_input_ids,finding_attention_mask,impression_attention_mask):

        output1 = self.model(input_ids=finding_input_ids, attention_mask =finding_attention_mask)
        output2 = self.model(input_ids=impression_input_ids, attention_mask= impression_attention_mask)

        outputs = torch.cat((output1.pooler_output, output2.pooler_output), 1)

        outputs = self.fc(outputs)
        return outputs
