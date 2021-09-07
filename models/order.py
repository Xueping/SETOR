from .modeling import BertEncoder, AttentionPooling, BertLayerNorm, PositionEmbeddings, \
    BertAttentionDag, BertIntermediateDag
import torch.nn as nn
import torch
import os
import logging
from models.anode import ODENet

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER
logger = logging.getLogger(__name__)


class ODEEmbeddings(nn.Module):
    def __init__(self, config):
        super(ODEEmbeddings, self).__init__()
        self.odeNet_los = ODENet(config.device, config.hidden_size, config.hidden_size)
        self.odeNet_interval = ODENet(config.device, config.hidden_size, config.hidden_size,
                                      output_dim=config.hidden_size, augment_dim=10, time_dependent=True)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_states):

        seq_length = input_states.size(1)
        integration_time = torch.linspace(0., 1., seq_length)
        y0 = input_states[:, 0, :]
        interval_embeddings = self.odeNet_interval(y0, integration_time).permute(1, 0, 2)

        los_embeddings = self.odeNet_los(input_states)
        embeddings = input_states + interval_embeddings + los_embeddings

        embeddings = self.dropout(embeddings)
        embeddings = self.LayerNorm(embeddings)
        return embeddings


class DAGAttention2D(nn.Module):
    def __init__(self, in_features, attention_dim_size):
        super(DAGAttention2D, self).__init__()
        self.attention_dim_size = attention_dim_size
        self.in_features = in_features
        self.linear1 = nn.Linear(in_features, attention_dim_size)
        self.linear2 = nn.Linear(attention_dim_size, 1)

    def forward(self, leaves, ancestors, mask=None):
        # concatenate the leaves and ancestors
        mask = mask.unsqueeze(2)
        x = torch.cat((leaves * mask, ancestors * mask), dim=-1)

        # Linear layer
        x = self.linear1(x)

        # relu activation
        x = torch.relu(x)

        # linear layer
        x = self.linear2(x)

        mask_attn = (1.0 - mask) * VERY_NEGATIVE_NUMBER
        x = x + mask_attn

        # softmax activation
        x = torch.softmax(x, dim=1)

        # weighted sum on ancestors
        x = (x * ancestors * mask).sum(dim=1)
        return x


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pre-trained models.
    """
    def __init__(self, config,  *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, config, state_dict=None, *inputs, **kwargs):
        print('parameters in inputs: ', *inputs)

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)

        if state_dict is None:
            weights_path = os.path.join(pretrained_model_name, 'pytorch_model.bin')
            state_dict = torch.load(weights_path).state_dict()

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))

        return model, missing_keys


class KnowledgeEncoder(nn.Module):
    """
    Only Sum up embeddings of codes and its knowledge
    """
    def __init__(self, config):
        super(KnowledgeEncoder, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.embed_dag = None

        self.dag_attention = DAGAttention2D(2 * config.hidden_size, config.hidden_size)
        self.encoder = BertEncoder(config)
        self.attention = BertAttentionDag(config)
        self.intermediate = BertIntermediateDag(config)

        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.pooling = AttentionPooling(config)

        self.embed_init = nn.Embedding(config.num_tree_nodes, config.hidden_size)
        self.embed_inputs = nn.Embedding(config.code_size, self.hidden_size)

    def forward(self, input_ids, code_mask=None, output_attentions=False):

        # for knowledge graph embedding
        leaves_emb = self.embed_init(self.config.leaves_list)
        ancestors_emb = self.embed_init(self.config.ancestors_list)
        dag_emb = self.dag_attention(leaves_emb, ancestors_emb, self.config.masks_list)
        padding = torch.zeros([1, self.hidden_size], dtype=torch.float32).to(self.config.device)
        dict_matrix = torch.cat([padding, dag_emb], dim=0)
        self.embed_dag = nn.Embedding.from_pretrained(dict_matrix, freeze=False)

        # inputs embedding
        input_tensor = self.embed_inputs(input_ids)  # bs, visit_len, code_len, embedding_dim
        input_shape = input_tensor.shape
        inputs = input_tensor.view(-1, input_shape[2], input_shape[3])  # bs * visit_len, code_len, embedding_dim

        # entity embedding
        input_tensor_dag = self.embed_dag(input_ids)
        # bs * visit_len, code_len, embedding_dim
        inputs_dag = input_tensor_dag.view(-1, input_tensor_dag.shape[2], input_tensor_dag.shape[3])

        inputs_mask = code_mask.view(-1, input_tensor_dag.shape[2])  # bs * visit_len, code_len

        # attention mask for encoder
        extended_attention_mask = inputs_mask.unsqueeze(1).unsqueeze(2)  # bs * visit_len,1,1 code_len
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * VERY_NEGATIVE_NUMBER

        hidden_states_output, hidden_states_dag_output, _, _ = \
            self.attention(inputs, extended_attention_mask, inputs_dag, None, None, output_attentions)

        intermediate_output = self.intermediate(hidden_states_output, hidden_states_dag_output)
        hidden_states = self.dense(intermediate_output)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + hidden_states_output + hidden_states_dag_output)

        # knowledge encoder
        visit_outputs, all_attentions = hidden_states, None

        # attention mask for pooling
        attention_mask = inputs_mask.unsqueeze(2)  # bs * visit_len,code_len,1
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * VERY_NEGATIVE_NUMBER

        # visit attention pooling
        visit_pooling = self.pooling(visit_outputs, attention_mask)
        visit_outs = visit_pooling.view(-1, input_tensor_dag.shape[1], input_tensor_dag.shape[3])  # bs, visit_len, embedding_dim

        return visit_outs, all_attentions


class NextDxPrediction(PreTrainedBertModel):
    def __init__(self, config):
        super(NextDxPrediction, self).__init__(config)

        self.encoder = KnowledgeEncoder(config)

        self.encoder_patient = BertEncoder(config)
        # self.position_embedding = PositionEmbeddings(config)
        self.position_embedding = ODEEmbeddings(config)

        self.classifier_patient = nn.Linear(config.hidden_size, config.num_ccs_classes)
        self.classifier_entity = nn.Linear(config.hidden_size, config.num_visit_classes)

        self.apply(self.init_bert_weights)

    def forward(self,
                input_ids,
                visit_mask=None,
                code_mask=None,
                labels_visit=None,
                output_attentions=False
                ):

        lengths = visit_mask.sum(axis=-1)
        outputs_visit, code_attentions = self.encoder(input_ids, code_mask, output_attentions)

        # add position embedding
        visit_outs = self.position_embedding(outputs_visit)

        extended_attention_mask = visit_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * VERY_NEGATIVE_NUMBER
        patient_outputs, visit_attentions = self.encoder_patient(visit_outs, extended_attention_mask,
                                                                 output_all_encoded_layers=False,
                                                                 output_attentions = output_attentions)

        prediction_scores_patient = self.classifier_patient(patient_outputs[-1])
        prediction_scores_patient = torch.sigmoid(prediction_scores_patient)

        if labels_visit is not None:
            logEps = 1e-8
            cross_entropy_patient = -(labels_visit * torch.log(prediction_scores_patient + logEps) +
                              (1. - labels_visit) * torch.log(1. - prediction_scores_patient + logEps))
            likelihood_patient = cross_entropy_patient.sum(axis=2).sum(axis=1) / lengths
            loss_patient = torch.mean(likelihood_patient)

            total_loss = loss_patient
            return total_loss
        else:
            return prediction_scores_patient, code_attentions, visit_attentions
