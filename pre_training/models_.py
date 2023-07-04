import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from transformers.file_utils import ModelOutput
from latest_transformer import Transformer

@dataclass
class SimpleOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


def sinusoidal_init(num_embeddings: int, embedding_dim: int):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / embedding_dim) for i in range(embedding_dim)]
        if pos != 0 else np.zeros(embedding_dim) for pos in range(num_embeddings)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)

class LongformerSiameseNetwork(nn.Module):

  def __init__(self):
    super(LongformerSiameseNetwork, self).__init__()

    self.model_layer = AutoModel.from_pretrained('allenai/longformer-base-4096')
    self.dense_layer = nn.Linear(768, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, input_ids1, input_ids2, attention_mask1, attention_mask2):

    output1 = self.model_layer.forward(input_ids1, attention_mask = attention_mask1)[0]
    output2 = self.model_layer.forward(input_ids2, attention_mask = attention_mask2)[0]

    avg_output_1 = torch.mean(output1, 1, True)
    avg_output_2 = torch.mean(output2, 1, True)
    diff = torch.sub(avg_output_1, avg_output_2, alpha = 1)

    output = self.dense_layer(diff)
    similarity = self.sigmoid(output)

    return similarity

class BertSiameseNetwork(nn.Module):

  def __init__(self):
    super(BertSiameseNetwork, self).__init__()

    self.model_layer = AutoModel.from_pretrained('bert-base-uncased')
    self.dense_layer = nn.Linear(768, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, input_ids1, input_ids2, attention_mask1, attention_mask2):

    output1 = self.model_layer.forward(input_ids1, attention_mask = attention_mask1)[0]
    output2 = self.model_layer.forward(input_ids2, attention_mask = attention_mask2)[0]

    avg_output_1 = torch.mean(output1, 1, True)
    avg_output_2 = torch.mean(output2, 1, True)
    diff = torch.sub(avg_output_1, avg_output_2, alpha = 1)

    output = self.dense_layer(diff)
    similarity = self.sigmoid(output)

    return similarity

class HierSiameseNetwork(nn.Module):

  def __init__(self, num_topics_each_level):
    super(HierSiameseNetwork, self).__init__()

    self.num_topics_each_level = num_topics_each_level
    self.model_layer = AutoModel.from_pretrained('bert-base-uncased')
    self.dense_layer = nn.Linear(768, 1)
    self.dense_layer1 = nn.Linear(768, self.num_topics_each_level[0])
    self.dense_layer2 = nn.Linear(768, self.num_topics_each_level[1])
    self.dense_layer3 = nn.Linear(768, self.num_topics_each_level[2])
    self.dense_layer4 = nn.Linear(768, self.num_topics_each_level[3])
    self.dense_layer5 = nn.Linear(768, self.num_topics_each_level[4])
    self.dense_layer6 = nn.Linear(768, self.num_topics_each_level[5])
    self.dense_layer7 = nn.Linear(768, self.num_topics_each_level[6])    
    self.sigmoid = nn.Sigmoid()
    #self.softmax = nn.Softmax()

  def forward(self, embedding1, embedding2):

    output1 = self.model_layer.forward(inputs_embeds = embedding1)[0]
    output2 = self.model_layer.forward(inputs_embeds = embedding2)[0]

    avg_output_1 = torch.mean(output1, 1, True)
    avg_output_2 = torch.mean(output2, 1, True)
    diff = torch.sub(avg_output_1, avg_output_2, alpha = 1)
    output = self.dense_layer(diff)
    similarity = self.sigmoid(output)

    heirarchy_out1_emb1 = self.dense_layer1(avg_output_1).squeeze()
    heirarchy_out2_emb1 = self.dense_layer2(avg_output_1).squeeze()
    heirarchy_out3_emb1 = self.dense_layer3(avg_output_1).squeeze()
    heirarchy_out4_emb1 = self.dense_layer4(avg_output_1).squeeze()
    heirarchy_out5_emb1 = self.dense_layer5(avg_output_1).squeeze()
    heirarchy_out6_emb1 = self.dense_layer6(avg_output_1).squeeze()
    heirarchy_out7_emb1 = self.dense_layer7(avg_output_1).squeeze()    

    heirarchy_out1_emb2 = self.dense_layer1(avg_output_2).squeeze()
    heirarchy_out2_emb2 = self.dense_layer2(avg_output_2).squeeze()
    heirarchy_out3_emb2 = self.dense_layer3(avg_output_2).squeeze()
    heirarchy_out4_emb2 = self.dense_layer4(avg_output_2).squeeze()
    heirarchy_out5_emb2 = self.dense_layer5(avg_output_2).squeeze()
    heirarchy_out6_emb2 = self.dense_layer6(avg_output_2).squeeze()
    heirarchy_out7_emb2 = self.dense_layer7(avg_output_2).squeeze()    

    #print(heirarchy_out5_emb2.shape, heirarchy_out5_emb1.shape)

    return similarity, heirarchy_out1_emb1, heirarchy_out2_emb1, heirarchy_out3_emb1, heirarchy_out4_emb1, heirarchy_out5_emb1, heirarchy_out6_emb1, heirarchy_out7_emb1,heirarchy_out1_emb2, heirarchy_out2_emb2, heirarchy_out3_emb2, heirarchy_out4_emb2, heirarchy_out5_emb2, heirarchy_out6_emb2, heirarchy_out7_emb2

class SiameseNetwork(nn.Module):

  def __init__(self):
    super(SiameseNetwork, self).__init__()

    self.model_layer = AutoModel.from_pretrained('bert-base-uncased')
    self.dense_layer = nn.Linear(768, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, embedding1, embedding2):

    output1 = self.model_layer.forward(inputs_embeds = embedding1)[0]
    output2 = self.model_layer.forward(inputs_embeds = embedding2)[0]
                
    avg_output_1 = torch.mean(output1, 1, True)
    avg_output_2 = torch.mean(output2, 1, True)
    diff = torch.sub(avg_output_1, avg_output_2, alpha = 1)
    output = self.dense_layer(diff)
    similarity = self.sigmoid(output)

    return similarity

class OnlyClassfnNetwork(nn.Module):

  def __init__(self, num_topics_each_level, model_path):
    super(OnlyClassfnNetwork, self).__init__()

    self.num_topics_each_level = num_topics_each_level
    self.model_layer = AutoModel.from_pretrained(model_path)
    # self.dense_layer = nn.Linear(768, 1)
    self.dense_layer1 = nn.Linear(768, self.num_topics_each_level[0])
    self.dense_layer2 = nn.Linear(768, self.num_topics_each_level[1])
    self.dense_layer3 = nn.Linear(768, self.num_topics_each_level[2])
    self.dense_layer4 = nn.Linear(768, self.num_topics_each_level[3])
    self.dense_layer5 = nn.Linear(768, self.num_topics_each_level[4])
    self.dense_layer6 = nn.Linear(768, self.num_topics_each_level[5])
    self.dense_layer7 = nn.Linear(768, self.num_topics_each_level[6])    
    # self.sigmoid = nn.Sigmoid()
    #self.softmax = nn.Softmax()

  def forward(self, embedding1, embedding2):

    output1 = self.model_layer.forward(inputs_embeds = embedding1)[0]
    output2 = self.model_layer.forward(inputs_embeds = embedding2)[0]

    avg_output_1 = torch.mean(output1, 1, True)
    avg_output_2 = torch.mean(output2, 1, True)
    # diff = torch.sub(avg_output_1, avg_output_2, alpha = 1)
    # output = self.dense_layer(diff)
    # similarity = self.sigmoid(output)

    heirarchy_out1_emb1 = self.dense_layer1(avg_output_1).squeeze()
    heirarchy_out2_emb1 = self.dense_layer2(avg_output_1).squeeze()
    heirarchy_out3_emb1 = self.dense_layer3(avg_output_1).squeeze()
    heirarchy_out4_emb1 = self.dense_layer4(avg_output_1).squeeze()
    heirarchy_out5_emb1 = self.dense_layer5(avg_output_1).squeeze()
    heirarchy_out6_emb1 = self.dense_layer6(avg_output_1).squeeze()
    heirarchy_out7_emb1 = self.dense_layer7(avg_output_1).squeeze()    

    heirarchy_out1_emb2 = self.dense_layer1(avg_output_2).squeeze()
    heirarchy_out2_emb2 = self.dense_layer2(avg_output_2).squeeze()
    heirarchy_out3_emb2 = self.dense_layer3(avg_output_2).squeeze()
    heirarchy_out4_emb2 = self.dense_layer4(avg_output_2).squeeze()
    heirarchy_out5_emb2 = self.dense_layer5(avg_output_2).squeeze()
    heirarchy_out6_emb2 = self.dense_layer6(avg_output_2).squeeze()
    heirarchy_out7_emb2 = self.dense_layer7(avg_output_2).squeeze()    

    #print(heirarchy_out5_emb2.shape, heirarchy_out5_emb1.shape)

    return heirarchy_out1_emb1, heirarchy_out2_emb1, heirarchy_out3_emb1, heirarchy_out4_emb1, heirarchy_out5_emb1, heirarchy_out6_emb1, heirarchy_out7_emb1,heirarchy_out1_emb2, heirarchy_out2_emb2, heirarchy_out3_emb2, heirarchy_out4_emb2, heirarchy_out5_emb2, heirarchy_out6_emb2, heirarchy_out7_emb2

class TripletNetwork(nn.Module):

  def __init__(self, model_path):
    super(TripletNetwork, self).__init__()

    self.model_layer = AutoModel.from_pretrained(model_path)

  def forward(self, embedding, embedding_pos, embedding_neg):

    output_anchor = self.model_layer.forward(inputs_embeds = embedding)[0]
    output_pos = self.model_layer.forward(inputs_embeds = embedding_pos)[0]
    output_neg = self.model_layer.forward(inputs_embeds = embedding_neg)[0]
                
    # avg_output_anchor = torch.mean(output_anchor, 1, True)
    # avg_output_pos = torch.mean(output_pos, 1, True)
    # avg_output_neg = torch.mean(output_neg, 1, True)

    return output_anchor, output_pos, output_neg

class QuadrupletNetwork(nn.Module):

  def __init__(self, model_path):
    super(QuadrupletNetwork, self).__init__()

    self.model_layer = AutoModel.from_pretrained(model_path)

  def forward(self, embedding, embedding_near_pos, embedding_far_pos, embedding_neg):

    output_anchor = self.model_layer.forward(inputs_embeds = embedding)[0]
    output_near_pos = self.model_layer.forward(inputs_embeds = embedding_near_pos)[0]
    output_far_pos = self.model_layer.forward(inputs_embeds = embedding_far_pos)[0]
    output_neg = self.model_layer.forward(inputs_embeds = embedding_neg)[0]
                
    # avg_output_anchor = torch.mean(output_anchor, 1, True)
    # avg_output_pos = torch.mean(output_pos, 1, True)
    # avg_output_neg = torch.mean(output_neg, 1, True)

    return output_anchor, output_near_pos, output_far_pos, output_neg

class HierTripletNetwork(nn.Module):

  def __init__(self, num_topics_each_level, model_path):
    super(HierTripletNetwork, self).__init__()

    self.num_topics_each_level = num_topics_each_level
    self.model_layer = AutoModel.from_pretrained(model_path)
    self.dense_layer = nn.Linear(768, 1)
    self.dense_layer1 = nn.Linear(768, self.num_topics_each_level[0])
    self.dense_layer2 = nn.Linear(768, self.num_topics_each_level[1])
    self.dense_layer3 = nn.Linear(768, self.num_topics_each_level[2])
    self.dense_layer4 = nn.Linear(768, self.num_topics_each_level[3])
    self.dense_layer5 = nn.Linear(768, self.num_topics_each_level[4])
    self.dense_layer6 = nn.Linear(768, self.num_topics_each_level[5])
    self.dense_layer7 = nn.Linear(768, self.num_topics_each_level[6])    
    self.sigmoid = nn.Sigmoid()
    #self.softmax = nn.Softmax()

  def forward(self, embedding, embedding_pos, embedding_neg):

    output_anchor = self.model_layer.forward(inputs_embeds = embedding)[0]
    output_pos = self.model_layer.forward(inputs_embeds = embedding_pos)[0]
    output_neg = self.model_layer.forward(inputs_embeds = embedding_neg)[0]

    avg_output_1 = torch.mean(output_anchor, 1, True)
    avg_output_2 = torch.mean(output_pos, 1, True)
    avg_output_3 = torch.mean(output_neg, 1, True)


    heirarchy_out1_emb1 = self.dense_layer1(avg_output_1).squeeze()
    heirarchy_out2_emb1 = self.dense_layer2(avg_output_1).squeeze()
    heirarchy_out3_emb1 = self.dense_layer3(avg_output_1).squeeze()
    heirarchy_out4_emb1 = self.dense_layer4(avg_output_1).squeeze()
    heirarchy_out5_emb1 = self.dense_layer5(avg_output_1).squeeze()
    heirarchy_out6_emb1 = self.dense_layer6(avg_output_1).squeeze()
    heirarchy_out7_emb1 = self.dense_layer7(avg_output_1).squeeze()    

    heirarchy_out1_emb2 = self.dense_layer1(avg_output_2).squeeze()
    heirarchy_out2_emb2 = self.dense_layer2(avg_output_2).squeeze()
    heirarchy_out3_emb2 = self.dense_layer3(avg_output_2).squeeze()
    heirarchy_out4_emb2 = self.dense_layer4(avg_output_2).squeeze()
    heirarchy_out5_emb2 = self.dense_layer5(avg_output_2).squeeze()
    heirarchy_out6_emb2 = self.dense_layer6(avg_output_2).squeeze()
    heirarchy_out7_emb2 = self.dense_layer7(avg_output_2).squeeze()    

    heirarchy_out1_emb3 = self.dense_layer1(avg_output_3).squeeze()
    heirarchy_out2_emb3 = self.dense_layer2(avg_output_3).squeeze()
    heirarchy_out3_emb3 = self.dense_layer3(avg_output_3).squeeze()
    heirarchy_out4_emb3 = self.dense_layer4(avg_output_3).squeeze()
    heirarchy_out5_emb3 = self.dense_layer5(avg_output_3).squeeze()
    heirarchy_out6_emb3 = self.dense_layer6(avg_output_3).squeeze()
    heirarchy_out7_emb3 = self.dense_layer7(avg_output_3).squeeze() 

    #print(heirarchy_out5_emb2.shape, heirarchy_out5_emb1.shape)

    return output_anchor, output_pos, output_neg, heirarchy_out1_emb1, heirarchy_out2_emb1, heirarchy_out3_emb1, heirarchy_out4_emb1, heirarchy_out5_emb1, heirarchy_out6_emb1, heirarchy_out7_emb1,heirarchy_out1_emb2, heirarchy_out2_emb2, heirarchy_out3_emb2, heirarchy_out4_emb2, heirarchy_out5_emb2, heirarchy_out6_emb2, heirarchy_out7_emb2, \
    heirarchy_out1_emb3, heirarchy_out2_emb3, heirarchy_out3_emb3, heirarchy_out4_emb3, heirarchy_out5_emb3, heirarchy_out6_emb3, heirarchy_out7_emb3

class HierQuadrupletNetwork(nn.Module):

  def __init__(self, num_topics_each_level, model_path):
    super(HierQuadrupletNetwork, self).__init__()

    self.num_topics_each_level = num_topics_each_level
    self.model_layer = AutoModel.from_pretrained(model_path)
    self.dense_layer = nn.Linear(768, 1)
    self.dense_layer1 = nn.Linear(768, self.num_topics_each_level[0])
    self.dense_layer2 = nn.Linear(768, self.num_topics_each_level[1])
    self.dense_layer3 = nn.Linear(768, self.num_topics_each_level[2])
    self.dense_layer4 = nn.Linear(768, self.num_topics_each_level[3])
    self.dense_layer5 = nn.Linear(768, self.num_topics_each_level[4])
    self.dense_layer6 = nn.Linear(768, self.num_topics_each_level[5])
    self.dense_layer7 = nn.Linear(768, self.num_topics_each_level[6])    
    self.sigmoid = nn.Sigmoid()
    #self.softmax = nn.Softmax()

  def forward(self, embedding, embedding_near_pos, embedding_far_pos, embedding_neg):

    output_anchor = self.model_layer.forward(inputs_embeds = embedding)[0]
    output_near_pos = self.model_layer.forward(inputs_embeds = embedding_near_pos)[0]
    output_far_pos = self.model_layer.forward(inputs_embeds = embedding_far_pos)[0]
    output_neg = self.model_layer.forward(inputs_embeds = embedding_neg)[0]

    avg_output_1 = torch.mean(output_anchor, 1, True)
    avg_output_2 = torch.mean(output_near_pos, 1, True)
    avg_output_3 = torch.mean(output_far_pos, 1, True)
    avg_output_4 = torch.mean(output_neg, 1, True)


    heirarchy_out1_emb1 = self.dense_layer1(avg_output_1).squeeze()
    heirarchy_out2_emb1 = self.dense_layer2(avg_output_1).squeeze()
    heirarchy_out3_emb1 = self.dense_layer3(avg_output_1).squeeze()
    heirarchy_out4_emb1 = self.dense_layer4(avg_output_1).squeeze()
    heirarchy_out5_emb1 = self.dense_layer5(avg_output_1).squeeze()
    heirarchy_out6_emb1 = self.dense_layer6(avg_output_1).squeeze()
    heirarchy_out7_emb1 = self.dense_layer7(avg_output_1).squeeze()    

    heirarchy_out1_emb2 = self.dense_layer1(avg_output_2).squeeze()
    heirarchy_out2_emb2 = self.dense_layer2(avg_output_2).squeeze()
    heirarchy_out3_emb2 = self.dense_layer3(avg_output_2).squeeze()
    heirarchy_out4_emb2 = self.dense_layer4(avg_output_2).squeeze()
    heirarchy_out5_emb2 = self.dense_layer5(avg_output_2).squeeze()
    heirarchy_out6_emb2 = self.dense_layer6(avg_output_2).squeeze()
    heirarchy_out7_emb2 = self.dense_layer7(avg_output_2).squeeze()    

    heirarchy_out1_emb3 = self.dense_layer1(avg_output_3).squeeze()
    heirarchy_out2_emb3 = self.dense_layer2(avg_output_3).squeeze()
    heirarchy_out3_emb3 = self.dense_layer3(avg_output_3).squeeze()
    heirarchy_out4_emb3 = self.dense_layer4(avg_output_3).squeeze()
    heirarchy_out5_emb3 = self.dense_layer5(avg_output_3).squeeze()
    heirarchy_out6_emb3 = self.dense_layer6(avg_output_3).squeeze()
    heirarchy_out7_emb3 = self.dense_layer7(avg_output_3).squeeze() 

    heirarchy_out1_emb4 = self.dense_layer1(avg_output_4).squeeze()
    heirarchy_out2_emb4 = self.dense_layer2(avg_output_4).squeeze()
    heirarchy_out3_emb4 = self.dense_layer3(avg_output_4).squeeze()
    heirarchy_out4_emb4 = self.dense_layer4(avg_output_4).squeeze()
    heirarchy_out5_emb4 = self.dense_layer5(avg_output_4).squeeze()
    heirarchy_out6_emb4 = self.dense_layer6(avg_output_4).squeeze()
    heirarchy_out7_emb4 = self.dense_layer7(avg_output_4).squeeze() 

    #print(heirarchy_out5_emb2.shape, heirarchy_out5_emb1.shape)

    return output_anchor, output_near_pos, output_far_pos, output_neg, heirarchy_out1_emb1, heirarchy_out2_emb1, heirarchy_out3_emb1, heirarchy_out4_emb1, heirarchy_out5_emb1, heirarchy_out6_emb1, heirarchy_out7_emb1,heirarchy_out1_emb2, heirarchy_out2_emb2, heirarchy_out3_emb2, heirarchy_out4_emb2, heirarchy_out5_emb2, heirarchy_out6_emb2, heirarchy_out7_emb2, \
    heirarchy_out1_emb3, heirarchy_out2_emb3, heirarchy_out3_emb3, heirarchy_out4_emb3, heirarchy_out5_emb3, heirarchy_out6_emb3, heirarchy_out7_emb3, \
    heirarchy_out1_emb4, heirarchy_out2_emb4, heirarchy_out3_emb4, heirarchy_out4_emb4, heirarchy_out5_emb4, heirarchy_out6_emb4, heirarchy_out7_emb4

class HierarchicalBert(nn.Module): # here Hierarchical means two stages fo transformer

    def __init__(self, encoder, max_segments=32, max_segment_length=128):
        super(HierarchicalBert, self).__init__()
        supported_models = ['bert', 'roberta', 'deberta']
        assert encoder.config.model_type in supported_models  # other model types are not supported so far
        # Pre-trained segment (token-wise) encoder, e.g., BERT
        self.encoder = encoder
        # Specs for the segment-wise encoder
        self.hidden_size = encoder.config.hidden_size
        self.max_segments = max_segments
        self.max_segment_length = max_segment_length
        # Init sinusoidal positional embeddings
        self.seg_pos_embeddings = nn.Embedding(max_segments + 1, encoder.config.hidden_size,
                                               padding_idx=0,
                                               _weight=sinusoidal_init(max_segments + 1, encoder.config.hidden_size))
        # Init segment-wise transformer-based encoder
        self.seg_encoder = Transformer(d_model=encoder.config.hidden_size,
                                          nhead=encoder.config.num_attention_heads,
                                          batch_first=True, dim_feedforward=encoder.config.intermediate_size,
                                          activation=encoder.config.hidden_act,
                                          dropout=encoder.config.hidden_dropout_prob,
                                          layer_norm_eps=encoder.config.layer_norm_eps,
                                          num_encoder_layers=2, num_decoder_layers=0).encoder

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        # Hypothetical Example
        # Batch of 4 documents: (batch_size, n_segments, max_segment_length) --> (4, 64, 128)
        # BERT-BASE encoder: 768 hidden units

        # Squash samples and segments into a single axis (batch_size * n_segments, max_segment_length) --> (256, 128)
        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1))
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1))
        if token_type_ids is not None:
            token_type_ids_reshape = token_type_ids.contiguous().view(-1, token_type_ids.size(-1))
        else:
            token_type_ids_reshape = None

        # Encode segments with BERT --> (256, 128, 768)
        encoder_outputs = self.encoder(input_ids=input_ids_reshape,
                                       attention_mask=attention_mask_reshape,
                                       token_type_ids=token_type_ids_reshape)[0]

        # Reshape back to (batch_size, n_segments, max_segment_length, output_size) --> (4, 64, 128, 768)
        encoder_outputs = encoder_outputs.contiguous().view(input_ids.size(0), self.max_segments,
                                                            self.max_segment_length,
                                                            self.hidden_size)

        # Gather CLS outputs per segment --> (4, 64, 768)
        encoder_outputs = encoder_outputs[:, :, 0]

        # Infer real segments, i.e., mask paddings
        seg_mask = (torch.sum(input_ids, 2) != 0).to(input_ids.dtype)
        # Infer and collect segment positional embeddings
        seg_positions = torch.arange(1, self.max_segments + 1).to(input_ids.device) * seg_mask
        # Add segment positional embeddings to segment inputs
        encoder_outputs += self.seg_pos_embeddings(seg_positions)

        # Encode segments with segment-wise transformer
        seg_encoder_outputs = self.seg_encoder(encoder_outputs)

        # Collect document representation
        outputs, _ = torch.max(seg_encoder_outputs, 1)

        return SimpleOutput(last_hidden_state=outputs, hidden_states=outputs)

class twostagesiamese(nn.Module):

  def __init__(self):
    super(twostagesiamese, self).__init__()
    bert = AutoModel.from_pretrained('bert-base-uncased')
    self.model_layer = HierarchicalBert(bert)
    self.dense_layer = nn.Linear(768, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, input_ids1, input_ids2, attention_mask1, attention_mask2, token_type_ids1, token_type_ids2):

    output1 = self.model_layer.forward(input_ids1, attention_mask1, token_type_ids1)[0]
    output2 = self.model_layer.forward(input_ids2, attention_mask2, token_type_ids2)[0]
    # print(output1.shape, output2.shape)

    # avg_output_1 = torch.mean(output1, 1, True)
    # avg_output_2 = torch.mean(output2, 1, True)
    diff = torch.sub(output1, output2, alpha = 1)
    # print(diff.shape)

    output = self.dense_layer(diff)
    similarity = self.sigmoid(output)

    return similarity


class HierTwoStageSiamese(nn.Module):

  def __init__(self, num_topics_each_level):
    super(HierTwoStageSiamese, self).__init__()
    bert = AutoModel.from_pretrained('bert-base-uncased')
    self.num_topics_each_level = num_topics_each_level
    self.model_layer = HierarchicalBert(bert)
    self.dense_layer = nn.Linear(768, 1)
    self.dense_layer1 = nn.Linear(768, self.num_topics_each_level[0])
    self.dense_layer2 = nn.Linear(768, self.num_topics_each_level[1])
    self.dense_layer3 = nn.Linear(768, self.num_topics_each_level[2])
    self.dense_layer4 = nn.Linear(768, self.num_topics_each_level[3])
    self.dense_layer5 = nn.Linear(768, self.num_topics_each_level[4])
    self.dense_layer6 = nn.Linear(768, self.num_topics_each_level[5])
    self.dense_layer7 = nn.Linear(768, self.num_topics_each_level[6])    
    self.sigmoid = nn.Sigmoid()
    #self.softmax = nn.Softmax()

  def forward(self, input_ids1, input_ids2, attention_mask1, attention_mask2, token_type_ids1, token_type_ids2):

    avg_output_1 = self.model_layer.forward(input_ids1, attention_mask1, token_type_ids1)[0]
    avg_output_2 = self.model_layer.forward(input_ids2, attention_mask2, token_type_ids2)[0]
    diff = torch.sub(avg_output_1, avg_output_2, alpha = 1)
    output = self.dense_layer(diff)
    similarity = self.sigmoid(output)

    heirarchy_out1_emb1 = self.dense_layer1(avg_output_1)#.squeeze()
    heirarchy_out2_emb1 = self.dense_layer2(avg_output_1)#.squeeze()
    heirarchy_out3_emb1 = self.dense_layer3(avg_output_1)#.squeeze()
    heirarchy_out4_emb1 = self.dense_layer4(avg_output_1)#.squeeze()
    heirarchy_out5_emb1 = self.dense_layer5(avg_output_1)#.squeeze()
    heirarchy_out6_emb1 = self.dense_layer6(avg_output_1)#.squeeze()
    heirarchy_out7_emb1 = self.dense_layer7(avg_output_1)#.squeeze()    

    heirarchy_out1_emb2 = self.dense_layer1(avg_output_2)#.squeeze()
    heirarchy_out2_emb2 = self.dense_layer2(avg_output_2)#.squeeze()
    heirarchy_out3_emb2 = self.dense_layer3(avg_output_2)#.squeeze()
    heirarchy_out4_emb2 = self.dense_layer4(avg_output_2)#.squeeze()
    heirarchy_out5_emb2 = self.dense_layer5(avg_output_2)#.squeeze()
    heirarchy_out6_emb2 = self.dense_layer6(avg_output_2)#.squeeze()
    heirarchy_out7_emb2 = self.dense_layer7(avg_output_2)#.squeeze()    

    #print(heirarchy_out5_emb2.shape, heirarchy_out5_emb1.shape)

    return similarity, heirarchy_out1_emb1, heirarchy_out2_emb1, heirarchy_out3_emb1, heirarchy_out4_emb1, heirarchy_out5_emb1, heirarchy_out6_emb1, heirarchy_out7_emb1,heirarchy_out1_emb2, heirarchy_out2_emb2, heirarchy_out3_emb2, heirarchy_out4_emb2, heirarchy_out5_emb2, heirarchy_out6_emb2, heirarchy_out7_emb2

class OnlyClassfnTwoStage(nn.Module):

  def __init__(self, num_topics_each_level, model_path='bert-base-uncased'):
    super(OnlyClassfnTwoStage, self).__init__()
    bert = AutoModel.from_pretrained(model_path)
    self.model_path = model_path
    self.num_topics_each_level = num_topics_each_level
    self.model_layer = HierarchicalBert(bert)
    # self.dense_layer = nn.Linear(768, 1)
    self.dense_layer1 = nn.Linear(768, self.num_topics_each_level[0])
    self.dense_layer2 = nn.Linear(768, self.num_topics_each_level[1])
    self.dense_layer3 = nn.Linear(768, self.num_topics_each_level[2])
    self.dense_layer4 = nn.Linear(768, self.num_topics_each_level[3])
    self.dense_layer5 = nn.Linear(768, self.num_topics_each_level[4])
    self.dense_layer6 = nn.Linear(768, self.num_topics_each_level[5])
    self.dense_layer7 = nn.Linear(768, self.num_topics_each_level[6])    
    # self.sigmoid = nn.Sigmoid()
    #self.softmax = nn.Softmax()

  def forward(self, input_ids1, input_ids2, attention_mask1, attention_mask2, token_type_ids1=None, token_type_ids2=None):
    if self.model_path == 'roberta':
      avg_output_1 = self.model_layer.forward(input_ids1, attention_mask1)[0]
      avg_output_2 = self.model_layer.forward(input_ids2, attention_mask2)[0]      
    else:
      avg_output_1 = self.model_layer.forward(input_ids1, attention_mask1, token_type_ids1)[0]
      avg_output_2 = self.model_layer.forward(input_ids2, attention_mask2, token_type_ids2)[0]
    # diff = torch.sub(avg_output_1, avg_output_2, alpha = 1)
    # output = self.dense_layer(diff)
    # similarity = self.sigmoid(output)

    heirarchy_out1_emb1 = self.dense_layer1(avg_output_1)#.squeeze()
    heirarchy_out2_emb1 = self.dense_layer2(avg_output_1)#.squeeze()
    heirarchy_out3_emb1 = self.dense_layer3(avg_output_1)#.squeeze()
    heirarchy_out4_emb1 = self.dense_layer4(avg_output_1)#.squeeze()
    heirarchy_out5_emb1 = self.dense_layer5(avg_output_1)#.squeeze()
    heirarchy_out6_emb1 = self.dense_layer6(avg_output_1)#.squeeze()
    heirarchy_out7_emb1 = self.dense_layer7(avg_output_1)#.squeeze()    

    heirarchy_out1_emb2 = self.dense_layer1(avg_output_2)#.squeeze()
    heirarchy_out2_emb2 = self.dense_layer2(avg_output_2)#.squeeze()
    heirarchy_out3_emb2 = self.dense_layer3(avg_output_2)#.squeeze()
    heirarchy_out4_emb2 = self.dense_layer4(avg_output_2)#.squeeze()
    heirarchy_out5_emb2 = self.dense_layer5(avg_output_2)#.squeeze()
    heirarchy_out6_emb2 = self.dense_layer6(avg_output_2)#.squeeze()
    heirarchy_out7_emb2 = self.dense_layer7(avg_output_2)#.squeeze()    

    #print(heirarchy_out5_emb2.shape, heirarchy_out5_emb1.shape)

    return heirarchy_out1_emb1, heirarchy_out2_emb1, heirarchy_out3_emb1, heirarchy_out4_emb1, heirarchy_out5_emb1, heirarchy_out6_emb1, heirarchy_out7_emb1,heirarchy_out1_emb2, heirarchy_out2_emb2, heirarchy_out3_emb2, heirarchy_out4_emb2, heirarchy_out5_emb2, heirarchy_out6_emb2, heirarchy_out7_emb2

class TwoStageTriplet(nn.Module):

  def __init__(self, model_path='huawei-noah/TinyBERT_General_6L_768D'):
    super(TwoStageTriplet, self).__init__()
    bert = AutoModel.from_pretrained(model_path)
    self.model_layer = HierarchicalBert(bert)

  def forward(self, input_ids1, input_ids2, input_ids3, attention_mask1, attention_mask2, attention_mask3, token_type_ids1=None, token_type_ids2=None, token_type_ids3=None):

    avg_output_1 = self.model_layer.forward(input_ids1, attention_mask1, token_type_ids1)[0]
    avg_output_2 = self.model_layer.forward(input_ids2, attention_mask2, token_type_ids2)[0]
    avg_output_3 = self.model_layer.forward(input_ids3, attention_mask3, token_type_ids3)[0]

    return avg_output_1, avg_output_2, avg_output_3

class TwoStageQuadruplet(nn.Module):

  def __init__(self, model_path='huawei-noah/TinyBERT_General_6L_768D'):
    super(TwoStageQuadruplet, self).__init__()
    bert = AutoModel.from_pretrained(model_path)
    self.model_layer = HierarchicalBert(bert)

  def forward(self, input_ids1, input_ids2, input_ids3, input_ids4, attention_mask1, attention_mask2, attention_mask3, attention_mask4, token_type_ids1=None, token_type_ids2=None, token_type_ids3=None, token_type_ids4=None):

    avg_output_1 = self.model_layer.forward(input_ids1, attention_mask1, token_type_ids1)[0]
    avg_output_2 = self.model_layer.forward(input_ids2, attention_mask2, token_type_ids2)[0]
    avg_output_3 = self.model_layer.forward(input_ids3, attention_mask3, token_type_ids3)[0]
    avg_output_4 = self.model_layer.forward(input_ids4, attention_mask4, token_type_ids4)[0]

    return avg_output_1, avg_output_2, avg_output_3, avg_output_4

class HierTwoStageTriplet(nn.Module):

  def __init__(self, num_topics_each_level, model_path='huawei-noah/TinyBERT_General_6L_768D'):
    super(HierTwoStageTriplet, self).__init__()
    bert = AutoModel.from_pretrained(model_path)
    self.num_topics_each_level = num_topics_each_level
    self.model_layer = HierarchicalBert(bert)
    # self.dense_layer = nn.Linear(768, 1)
    self.dense_layer1 = nn.Linear(768, self.num_topics_each_level[0])
    self.dense_layer2 = nn.Linear(768, self.num_topics_each_level[1])
    self.dense_layer3 = nn.Linear(768, self.num_topics_each_level[2])
    self.dense_layer4 = nn.Linear(768, self.num_topics_each_level[3])
    self.dense_layer5 = nn.Linear(768, self.num_topics_each_level[4])
    self.dense_layer6 = nn.Linear(768, self.num_topics_each_level[5])
    self.dense_layer7 = nn.Linear(768, self.num_topics_each_level[6])    
    # self.sigmoid = nn.Sigmoid()
    #self.softmax = nn.Softmax()

  def forward(self, input_ids1, input_ids2, input_ids3, attention_mask1, attention_mask2, attention_mask3, token_type_ids1=None, token_type_ids2=None, token_type_ids3=None):

    avg_output_1 = self.model_layer.forward(input_ids1, attention_mask1, token_type_ids1)[0]
    avg_output_2 = self.model_layer.forward(input_ids2, attention_mask2, token_type_ids2)[0]
    avg_output_3 = self.model_layer.forward(input_ids3, attention_mask3, token_type_ids3)[0]
    # diff = torch.sub(avg_output_1, avg_output_2, alpha = 1)
    # output = self.dense_layer(diff)
    # similarity = self.sigmoid(output)

    heirarchy_out1_emb1 = self.dense_layer1(avg_output_1)#.squeeze()
    heirarchy_out2_emb1 = self.dense_layer2(avg_output_1)#.squeeze()
    heirarchy_out3_emb1 = self.dense_layer3(avg_output_1)#.squeeze()
    heirarchy_out4_emb1 = self.dense_layer4(avg_output_1)#.squeeze()
    heirarchy_out5_emb1 = self.dense_layer5(avg_output_1)#.squeeze()
    heirarchy_out6_emb1 = self.dense_layer6(avg_output_1)#.squeeze()
    heirarchy_out7_emb1 = self.dense_layer7(avg_output_1)#.squeeze()    

    heirarchy_out1_emb2 = self.dense_layer1(avg_output_2)#.squeeze()
    heirarchy_out2_emb2 = self.dense_layer2(avg_output_2)#.squeeze()
    heirarchy_out3_emb2 = self.dense_layer3(avg_output_2)#.squeeze()
    heirarchy_out4_emb2 = self.dense_layer4(avg_output_2)#.squeeze()
    heirarchy_out5_emb2 = self.dense_layer5(avg_output_2)#.squeeze()
    heirarchy_out6_emb2 = self.dense_layer6(avg_output_2)#.squeeze()
    heirarchy_out7_emb2 = self.dense_layer7(avg_output_2)#.squeeze() 

    heirarchy_out1_emb3 = self.dense_layer1(avg_output_3)#.squeeze()
    heirarchy_out2_emb3 = self.dense_layer2(avg_output_3)#.squeeze()
    heirarchy_out3_emb3 = self.dense_layer3(avg_output_3)#.squeeze()
    heirarchy_out4_emb3 = self.dense_layer4(avg_output_3)#.squeeze()
    heirarchy_out5_emb3 = self.dense_layer5(avg_output_3)#.squeeze()
    heirarchy_out6_emb3 = self.dense_layer6(avg_output_3)#.squeeze()
    heirarchy_out7_emb3 = self.dense_layer7(avg_output_3)#.squeeze()        

    #print(heirarchy_out5_emb2.shape, heirarchy_out5_emb1.shape)

    return avg_output_1, avg_output_2, avg_output_3, heirarchy_out1_emb1, heirarchy_out2_emb1, heirarchy_out3_emb1, heirarchy_out4_emb1, heirarchy_out5_emb1, heirarchy_out6_emb1, heirarchy_out7_emb1, heirarchy_out1_emb2, heirarchy_out2_emb2, heirarchy_out3_emb2, heirarchy_out4_emb2, heirarchy_out5_emb2, heirarchy_out6_emb2, heirarchy_out7_emb2, \
            heirarchy_out1_emb3, heirarchy_out2_emb3, heirarchy_out3_emb3, heirarchy_out4_emb3, heirarchy_out5_emb3, heirarchy_out6_emb3, heirarchy_out7_emb3

class HierTwoStageQuadruplet(nn.Module):

  def __init__(self, num_topics_each_level, model_path='huawei-noah/TinyBERT_General_6L_768D'):
    super(HierTwoStageQuadruplet, self).__init__()
    bert = AutoModel.from_pretrained(model_path)
    self.num_topics_each_level = num_topics_each_level
    self.model_layer = HierarchicalBert(bert)
    # self.dense_layer = nn.Linear(768, 1)
    self.dense_layer1 = nn.Linear(768, self.num_topics_each_level[0])
    self.dense_layer2 = nn.Linear(768, self.num_topics_each_level[1])
    self.dense_layer3 = nn.Linear(768, self.num_topics_each_level[2])
    self.dense_layer4 = nn.Linear(768, self.num_topics_each_level[3])
    self.dense_layer5 = nn.Linear(768, self.num_topics_each_level[4])
    self.dense_layer6 = nn.Linear(768, self.num_topics_each_level[5])
    self.dense_layer7 = nn.Linear(768, self.num_topics_each_level[6])    
    # self.sigmoid = nn.Sigmoid()
    #self.softmax = nn.Softmax()

  def forward(self, input_ids1, input_ids2, input_ids3, input_ids4, attention_mask1, attention_mask2, attention_mask3, attention_mask4, token_type_ids1=None, token_type_ids2=None, token_type_ids3=None, token_type_ids4=None):

    avg_output_1 = self.model_layer.forward(input_ids1, attention_mask1, token_type_ids1)[0]
    avg_output_2 = self.model_layer.forward(input_ids2, attention_mask2, token_type_ids2)[0]
    avg_output_3 = self.model_layer.forward(input_ids3, attention_mask3, token_type_ids3)[0]
    avg_output_4 = self.model_layer.forward(input_ids4, attention_mask4, token_type_ids4)[0]
    # diff = torch.sub(avg_output_1, avg_output_2, alpha = 1)
    # output = self.dense_layer(diff)
    # similarity = self.sigmoid(output)

    heirarchy_out1_emb1 = self.dense_layer1(avg_output_1)#.squeeze()
    heirarchy_out2_emb1 = self.dense_layer2(avg_output_1)#.squeeze()
    heirarchy_out3_emb1 = self.dense_layer3(avg_output_1)#.squeeze()
    heirarchy_out4_emb1 = self.dense_layer4(avg_output_1)#.squeeze()
    heirarchy_out5_emb1 = self.dense_layer5(avg_output_1)#.squeeze()
    heirarchy_out6_emb1 = self.dense_layer6(avg_output_1)#.squeeze()
    heirarchy_out7_emb1 = self.dense_layer7(avg_output_1)#.squeeze()    

    heirarchy_out1_emb2 = self.dense_layer1(avg_output_2)#.squeeze()
    heirarchy_out2_emb2 = self.dense_layer2(avg_output_2)#.squeeze()
    heirarchy_out3_emb2 = self.dense_layer3(avg_output_2)#.squeeze()
    heirarchy_out4_emb2 = self.dense_layer4(avg_output_2)#.squeeze()
    heirarchy_out5_emb2 = self.dense_layer5(avg_output_2)#.squeeze()
    heirarchy_out6_emb2 = self.dense_layer6(avg_output_2)#.squeeze()
    heirarchy_out7_emb2 = self.dense_layer7(avg_output_2)#.squeeze() 

    heirarchy_out1_emb3 = self.dense_layer1(avg_output_3)#.squeeze()
    heirarchy_out2_emb3 = self.dense_layer2(avg_output_3)#.squeeze()
    heirarchy_out3_emb3 = self.dense_layer3(avg_output_3)#.squeeze()
    heirarchy_out4_emb3 = self.dense_layer4(avg_output_3)#.squeeze()
    heirarchy_out5_emb3 = self.dense_layer5(avg_output_3)#.squeeze()
    heirarchy_out6_emb3 = self.dense_layer6(avg_output_3)#.squeeze()
    heirarchy_out7_emb3 = self.dense_layer7(avg_output_3)#.squeeze() 

    heirarchy_out1_emb4 = self.dense_layer1(avg_output_4)#.squeeze()
    heirarchy_out2_emb4 = self.dense_layer2(avg_output_4)#.squeeze()
    heirarchy_out3_emb4 = self.dense_layer3(avg_output_4)#.squeeze()
    heirarchy_out4_emb4 = self.dense_layer4(avg_output_4)#.squeeze()
    heirarchy_out5_emb4 = self.dense_layer5(avg_output_4)#.squeeze()
    heirarchy_out6_emb4 = self.dense_layer6(avg_output_4)#.squeeze()
    heirarchy_out7_emb4 = self.dense_layer7(avg_output_4)#.squeeze()        

    #print(heirarchy_out5_emb2.shape, heirarchy_out5_emb1.shape)

    return avg_output_1, avg_output_2, avg_output_3, avg_output_4, heirarchy_out1_emb1, heirarchy_out2_emb1, heirarchy_out3_emb1, heirarchy_out4_emb1, heirarchy_out5_emb1, heirarchy_out6_emb1, heirarchy_out7_emb1, heirarchy_out1_emb2, heirarchy_out2_emb2, heirarchy_out3_emb2, heirarchy_out4_emb2, heirarchy_out5_emb2, heirarchy_out6_emb2, heirarchy_out7_emb2, \
            heirarchy_out1_emb3, heirarchy_out2_emb3, heirarchy_out3_emb3, heirarchy_out4_emb3, heirarchy_out5_emb3, heirarchy_out6_emb3, heirarchy_out7_emb3, \
            heirarchy_out1_emb4, heirarchy_out2_emb4, heirarchy_out3_emb4, heirarchy_out4_emb4, heirarchy_out5_emb4, heirarchy_out6_emb4, heirarchy_out7_emb4

class SiameseLargeNetwork(nn.Module):

  def __init__(self):
    super(SiameseLargeNetwork, self).__init__()

    self.model_layer = AutoModel.from_pretrained('bert-large-uncased')
    self.dense_layer = nn.Linear(1024, 1)
    self.sigmoid = nn.Sigmoid()

  def forward(self, embedding1, embedding2):

    output1 = self.model_layer.forward(inputs_embeds = embedding1)[0]
    output2 = self.model_layer.forward(inputs_embeds = embedding2)[0]
                
    avg_output_1 = torch.mean(output1, 1, True)
    avg_output_2 = torch.mean(output2, 1, True)
    diff = torch.sub(avg_output_1, avg_output_2, alpha = 1)
    output = self.dense_layer(diff)
    similarity = self.sigmoid(output)

    return similarity