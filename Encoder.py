import torch
from torch import nn
from transformers import AutoModel


class Encoder(nn.Module):

    def __init__(self, feature_base_dim):

        super(Encoder, self).__init__()

        self.feature_base_dim = feature_base_dim
        self.batch_first = True

        self.sentence_bert = None

        self.model = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-base').to(device=self.device)

        self.fc = nn.Sequential(
            # nn.Linear(2 * self.hidden_size * self.num_dir, self.feature_base_dim, bias=False),
            nn.Linear(1536, self.feature_base_dim),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x1, x2, hidden=None):

        output1 = self.model(input_ids=x1['input_ids'], attention_mask=x1['attention_mask'])
        output2 = self.model(input_ids=x1['input_ids'], attention_mask=x1['attention_mask'])

        outputs = torch.cat((output1, output2), 1)

        outputs = self.fc(outputs)
        return outputs, hidden

