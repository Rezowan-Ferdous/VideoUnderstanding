from torch import nn

class ProjectionHead(nn.Module):
    """
    The projection head that maps visual/textual features
    to a shared embedding space.
    """
    def __init__(self, embedding_dim: int, projection_dim: int, dropout: float):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected # Residual connection
        x = self.layer_norm(x)
        return x
