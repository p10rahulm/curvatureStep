import torch
import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer, AutoModel


class BERTClassifier(nn.Module):
    def __init__(self, num_classes=4, freeze_bert=False):
        super(BERTClassifier, self).__init__()
        # Create a BERT configuration from scratch
        config = BertConfig()

        # Initialize the BERT model without pre-trained weights
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # CLS token's output
        x = self.dropout(pooled_output)
        x = self.linear(x)
        return self.softmax(x)




class TinyBERTClassifier(nn.Module):
    def __init__(self, num_classes=4, freeze_bert=False):
        super(TinyBERTClassifier, self).__init__()
        # Create a configuration for TinyBERT
        config = BertConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            max_position_embeddings=512,
            vocab_size=30522,
            type_vocab_size=2,
            hidden_act='gelu',
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0
        )

        # Initialize the BERT model without pre-trained weights
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # CLS token's output
        x = self.dropout(pooled_output)
        x = self.linear(x)
        return self.softmax(x)


class TinyBERTClassifierND(nn.Module):
    def __init__(self, num_classes=4, freeze_bert=False):
        super(TinyBERTClassifier, self).__init__()
        # Create a configuration for TinyBERT
        config = BertConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=512,
            max_position_embeddings=512,
            vocab_size=30522,
            type_vocab_size=2,
            hidden_act='gelu',
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            pad_token_id=0
        )

        # Initialize the BERT model without pre-trained weights
        self.bert = BertModel(config)

        self.dropout = nn.Dropout(0.01)
        self.linear = nn.Linear(config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # CLS token's output
        x = self.dropout(pooled_output)
        x = self.linear(x)
        return self.softmax(x)

class PretrainedBERTClassifier(nn.Module):
    def __init__(self, num_classes=2, freeze_bert=True):
        super(PretrainedBERTClassifier, self).__init__()
        # Initialize the BERT model with pre-trained weights
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Freeze BERT layers if required
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # CLS token's output
        x = self.dropout(pooled_output)
        x = self.linear(x)
        return self.softmax(x)

