from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import json
import math
import logging
import torch
from torch import nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class AttentionPooling(nn.Module):
    def __init__(self, config):
        super(AttentionPooling, self).__init__()
        self.config = config
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_tensor, pooling_mask):
        pooling_score = self.linear1(input_tensor)
        # pooling_score = ACT2FN['relu'](pooling_score)
        pooling_score = ACT2FN[self.config.hidden_act](pooling_score)

        pooling_score = self.linear2(pooling_score)

        pooling_score += pooling_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=1)(pooling_score)

        attention_output = (attention_probs * input_tensor).sum(dim=1)
        return attention_output


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_types=None):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_types = layer_types
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.code_size, config.hidden_size)
        if config.position_embedding == 'True':
            self.is_position_embedding = True
        else:
            self.is_position_embedding = False
        if self.is_position_embedding:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        if self.is_position_embedding:
            seq_length = input_ids.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            position_embeddings = self.position_embeddings(position_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if self.is_position_embedding:
            embeddings = words_embeddings + position_embeddings + token_type_embeddings
        else:
            embeddings = words_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class PositionEmbeddings(nn.Module):
    def __init__(self, config):
        super(PositionEmbeddings, self).__init__()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_states):

        seq_length = input_states.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_states.device)
        position_ids = position_ids.unsqueeze(0).repeat(input_states.size(0),1)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = input_states + position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.alpha = config.alpha
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.cnn = nn.Conv2d(self.num_attention_heads, self.num_attention_heads, 1, stride=1, padding=0)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, previous_attention, output_attentions=False):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
        # print(attention_scores[0][0][0][0])
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # residual connect the previous attentions.
        if previous_attention is not None and self.alpha > 0:
            # 1*1 CNN for previous attentions
            attention_probs = self.alpha * self.cnn(previous_attention) + (1 - self.alpha) * attention_probs
            # # residual connection with previous attention
            # attention_probs = self.alpha * previous_attention + (1 - self.alpha) * attention_probs
            attention_probs = nn.Softmax(dim=-1)(attention_probs)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, previous_attention, output_attentions=False):
        self_output = self.self(input_tensor, attention_mask, previous_attention, output_attentions)
        attention_output = self.output(self_output[0], input_tensor)
        outputs = (attention_output,) + self_output[1:]  # add attentions if we output them
        return outputs


class BertAttentionDag(nn.Module):
    def __init__(self, config):
        super(BertAttentionDag, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

        self.self_dag = BertSelfAttention(config)
        self.output_dag = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, input_tensor_dag, previous_attention,
                previous_attention_dag, output_attentions=False):
        self_output = self.self(input_tensor, attention_mask, previous_attention, output_attentions)
        self_output_dag = self.self_dag(input_tensor_dag, attention_mask, previous_attention_dag, output_attentions)

        attention_output = self.output(self_output[0], input_tensor)
        attention_output_dag = self.output_dag(self_output_dag[0], input_tensor_dag)
        outputs = (attention_output, attention_output_dag,) + self_output[1:] + self_output_dag[1:]
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertIntermediateDag(nn.Module):
    def __init__(self, config):
        super(BertIntermediateDag, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense_dag = nn.Linear(config.hidden_size, config.intermediate_size)

        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states, hidden_states_dag):
        hidden_states_ = self.dense(hidden_states)
        hidden_states_dag_ = self.dense_dag(hidden_states_dag.float())
        hidden_states = self.intermediate_act_fn(hidden_states_+hidden_states_dag_)

        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class BertOutputDag(nn.Module):
    def __init__(self, config):
        super(BertOutputDag, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense_dag = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.LayerNorm_dag = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states_, input_tensor, input_tensor_dag):
        hidden_states = self.dense(hidden_states_)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        hidden_states_dag = self.dense_dag(hidden_states_)
        hidden_states_dag = self.dropout(hidden_states_dag)
        hidden_states_dag = self.LayerNorm_dag(hidden_states_dag + input_tensor_dag)

        return hidden_states, hidden_states_dag


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, previous_attention, output_attentions=False):
        attention_output = self.attention(hidden_states, attention_mask, previous_attention, output_attentions)
        outputs = attention_output[1:]  # add self attentions if we output attention weights
        intermediate_output = self.intermediate(attention_output[0])
        layer_output = self.output(intermediate_output, attention_output[0])

        outputs = (layer_output,) + outputs
        return outputs


class BertLayerDag(nn.Module):
    def __init__(self, config):
        super(BertLayerDag, self).__init__()
        self.attention = BertAttentionDag(config)
        self.intermediate = BertIntermediateDag(config)
        self.output = BertOutputDag(config)

    def forward(self, hidden_states, attention_mask, hidden_states_dag, previous_attention, previous_attention_dag,
                output_attentions=False):
        hidden_states_output, hidden_states_dag_output, attention_output, attention_output_dag = \
            self.attention(hidden_states, attention_mask, hidden_states_dag, previous_attention,
                           previous_attention_dag, output_attentions)

        intermediate_output = self.intermediate(hidden_states_output, hidden_states_dag_output)
        layer_output, layer_output_dag = \
            self.output(intermediate_output, hidden_states_output, hidden_states_dag_output)
        return layer_output, layer_output_dag, attention_output, attention_output_dag


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        layers = []
        for _ in range(config.num_hidden_layers):
            layers.append(copy.deepcopy(layer))
        self.layer = nn.ModuleList(layers)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True,
                output_attentions=False, previous_attention=None):
        all_encoder_layers = []
        all_attentions = () if output_attentions else None
        for layer_module in self.layer:
            # print(hidden_states[0][0][0])
            layer_outputs = layer_module(hidden_states, attention_mask, previous_attention, output_attentions)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)

            hidden_states = layer_outputs[0]
            previous_attention = layer_outputs[1]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers, all_attentions


class BertEncoderDag(nn.Module):
    def __init__(self, config):
        super(BertEncoderDag, self).__init__()
        layer = BertLayerDag(config)
        layers = []
        for _ in range(config.num_hidden_layers):
            layers.append(copy.deepcopy(layer))
        self.layer = nn.ModuleList(layers)

    def forward(self, hidden_states, attention_mask, dag_inputs, output_all_encoded_layers=True,
                output_attentions=False, previous_attention=None, previous_attention_dag=None):
        all_encoder_layers = []
        all_encoder_dags = []
        all_attentions = () if output_attentions else None
        for layer_module in self.layer:
            hidden_states, dag_inputs, attentions, attentions_dag = layer_module(hidden_states, attention_mask,
                                                                                 dag_inputs, previous_attention,
                                                                                 previous_attention_dag,
                                                                                 output_attentions)
            previous_attention = attentions
            previous_attention_dag = attentions_dag
            if output_attentions:
                all_attentions = all_attentions + (attentions,)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                all_encoder_dags.append(dag_inputs)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            all_encoder_dags.append(dag_inputs)
        return all_encoder_layers, all_encoder_dags, all_attentions

