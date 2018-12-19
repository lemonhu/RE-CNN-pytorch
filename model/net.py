"""Define the neural network, loss function"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, data_loader, params):
        super(Net, self).__init__()
        # loading embedding vectors of dataset
        embedding_vectors = data_loader.get_loaded_embedding_vectors()
        # word and position embedding layer
        self.word_embedding = nn.Embedding.from_pretrained(embeddings=embedding_vectors, freeze=False)
        self.pos1_embedding = nn.Embedding(params.pos_dis_limit * 2 + 3, params.pos_emb_dim)
        self.pos2_embedding = nn.Embedding(params.pos_dis_limit * 2 + 3, params.pos_emb_dim)

        # dropout layer
        self.dropout = nn.Dropout(params.dropout_ratio)

        feature_dim = params.word_emb_dim + params.pos_emb_dim * 2
        # encode sentence level features via cnn
        self.covns = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=params.filter_num,
                                              kernel_size=(k, feature_dim),
                                              padding=0) for k in params.filters]
                                   )

        filter_dim = params.filter_num * len(params.filters)
        labels_num = len(data_loader.label2idx)
        # output layer
        self.linear = nn.Linear(filter_dim, labels_num)

        self.loss = nn.CrossEntropyLoss()

        if params.gpu >= 0:
            self.cuda(device=params.gpu)

    def forward(self, x):
        batch_sents = x['sents']
        batch_pos1s = x['pos1s']
        batch_pos2s = x['pos2s']

        word_embs = self.word_embedding(batch_sents)
        pos1_embs = self.pos1_embedding(batch_pos1s)
        pos2_embs = self.pos2_embedding(batch_pos2s)

        input_feature = torch.cat([word_embs, pos1_embs, pos2_embs], dim=2)  # batch_size x batch_max_len x feature_dim
        x = input_feature.unsqueeze(1)  # batch_size x 1 x batch_max_len x feature_dim
        
        x = self.dropout(x)
        
        x = [torch.tanh(conv(x)).squeeze(3) for conv in self.covns]  # x[idx]: batch_size x batch_max_len x (batch_max_len-knernel_size+1)
        
        x = [F.max_pool1d(i, kernel_size=i.size(2)).squeeze(2) for i in x]  # x[idx]: batch_size x filter_num
        sentence_features = torch.cat(x, dim=1)  # batch_size x (filter_num * len(filters))

        x = self.dropout(sentence_features)

        x = self.linear(x)

        return x

