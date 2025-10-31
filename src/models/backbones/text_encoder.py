from torch import nn
from transformers import AutoModel, DistilBertModel, DistilBertConfig

class HuggingFaceTextEncoder(nn.Module):
    """
    A wrapper for a Hugging Face text model (e.g., DistilBERT, RoBERTa).
    """
    def __init__(self, model_name="distilbert-base-uncased", pretrained=True, trainable=False):
        super().__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            # Example for DistilBERT if not pretrained
            config = DistilBertConfig()
            self.model = DistilBertModel(config)
            
        # Freeze parameters
        for param in self.model.parameters():
            param.requires_grad = trainable

        # We use the [CLS] token representation as the sentence embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        # Return the [CLS] token's embedding
        return last_hidden_state[:, self.target_token_idx, :]



class TextEncoder(nn.Module):
    """
    A wrapper for a Hugging Face text model (e.g., DistilBERT).
    """
    def __init__(self, model_name: str, pretrained: bool = True, trainable: bool = True):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
            
        # Freeze parameters if not trainable
        for param in self.model.parameters():
            param.requires_grad = trainable

        # We use the [CLS] token representation
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        # Return the [CLS] token's embedding
        return last_hidden_state[:, self.target_token_idx, :]
    

    