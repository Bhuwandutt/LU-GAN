import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
# from Attention import AttentionWordRNN, AttentionSentRNN
from transformers import AutoTokenizer, AutoModel



class Encoder(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_size,
                 hidden_size,
                 max_len,
                 feature_base_dim,
                 n_layers=1,
                 dropout=0.5,
                 bi_direction=True):

        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embedding_size
        self.max_sent_len = max_len
        self.feature_base_dim = feature_base_dim
        self.num_dir = 2 if bi_direction else 1
        self.batch_first = True

        self.sentence_bert = None 
        self.tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-base')
        self.model = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-base')

        self.sentence_bert = self.model


        

        # self.word_RNN1 = AttentionWordRNN(num_tokens=self.vocab_size,
        #                                   embed_size=self.embed_size,
        #                                   word_gru_hidden=self.hidden_size,
        #                                   bidirectional=True,
        #                                   dropout=dropout,
        #                                   batch_first=self.batch_first)
        # # Sentence-Level LSTM
        # self.sentence_RNN1 = AttentionSentRNN(sent_gru_hidden=self.hidden_size,
        #                                       word_gru_hidden=self.hidden_size,
        #                                       feature_base_dim=self.feature_base_dim,
        #                                       bidirectional=bi_direction,
        #                                       dropout=dropout,
        #                                       batch_first=self.batch_first)
        # # Word-Level LSTM
        # self.word_RNN2 = AttentionWordRNN(num_tokens=self.vocab_size,
        #                                   embed_size=self.embed_size,
        #                                   word_gru_hidden=self.hidden_size,
        #                                   bidirectional=bi_direction,
        #                                   dropout=dropout,
        #                                   batch_first=self.batch_first)
        # # Sentence-Level LSTM
        # # self.embedding = LinkBERT(sentence to create embedding)

        # self.sentence_RNN2 = AttentionSentRNN(sent_gru_hidden=self.hidden_size,
        #                                       word_gru_hidden=self.hidden_size,
        #                                       feature_base_dim=self.feature_base_dim,
        #                                       bidirectional=bi_direction,
        #                                       dropout=dropout,
        #                                       batch_first=self.batch_first)

        # # -> ngf x 1 x 1
        self.fc = nn.Sequential(
            nn.Linear(2 * self.hidden_size * self.num_dir, self.feature_base_dim, bias=False),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x1, x2, hidden=None):
        outputs1 = self.forward_once1(x1)
        outputs2 = self.forward_once1(x2)

        outputs = torch.cat((outputs1, outputs2), 1)

        #print(f"Output Size :- {outputs.size()}")

        outputs = self.fc(outputs)
        return outputs, hidden

    def forward_once1(self, x, state_word=None):
        # print(x.shape)
        batch, sent_len = x.shape

        x = x.view(batch * sent_len, -1)
        
        #word_embed, state_word, _ = self.word_RNN1(x)
        word_emebd_tokens = self.tokenizer(
                                           x,
                                           padding='longest',
                                           max_length=self.max_sent_len,
                                           truncation=True,
                                           return_tensors="pt",  # Returns torch.tensor instead of python integers
                                          )
        input_ids = word_emebd_tokens.input_ids.to(device='cuda')
        attention_mask = word_emebd_tokens.attention_mask.to(device='cuda')
        
        with torch.no_grad():
          t5_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
          final_encoding = t5_output.last_hidden_state.detach()
      
        return final_encoding

        
        # all_word_embed = word_embed.view(batch, sent_len, -1)
        # sent_embed, state_sent, _ = self.sentence_RNN1(all_word_embed)
        # print(f"Sent Embed (forward 1)Size:- {sent_embed.size()}")

        

    # def forward_once2(self, x, state_word=None):
    #     batch, sent_len = x.shape
    #     x = x.view(batch * sent_len, -1)
    #     word_embed, state_word, _ = self.word_RNN2(x)
    #     all_word_embed = word_embed.view(batch, sent_len, -1)
    #     sent_embed, state_sent, _ = self.sentence_RNN2(all_word_embed)
    #     #print(f"Sent Embed Size:- {sent_embed.size()}")
    #     return sent_embed
