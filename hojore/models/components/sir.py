import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(SelfAttentionTransformer, self).__init__()
        self.self_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.layers = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim) for _ in range(num_layers)]
        )
        
    def forward(self, x):
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, feature_dim] -> [seq_len, batch_size, feature_dim]
        for layer in self.layers:
            x = layer(x)
        return x.permute(1, 0, 2)  # [seq_len, batch_size, feature_dim] -> [batch_size, seq_len, feature_dim]

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(TransformerDecoder, self).__init__()
        self.transformer_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim), num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, feature_dim] -> [seq_len, batch_size, feature_dim]
        x = self.transformer_decoder(x, x)
        x = self.fc(x)
        return x.permute(1, 0, 2)  # [seq_len, batch_size, feature_dim] -> [batch_size, seq_len, feature_dim]

class TemporalDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(TemporalDecoder, self).__init__()
        self.temporal_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim), num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        x = x.permute(1, 0, 2)  # [batch_size, seq_len, feature_dim] -> [seq_len, batch_size, feature_dim]
        x = self.temporal_decoder(x, x)
        x = self.fc(x)
        return x.permute(1, 0, 2)  # [seq_len, batch_size, feature_dim] -> [batch_size, seq_len, feature_dim]

class SIR(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(SIR, self).__init__()
        self.self_attention_transformer = SelfAttentionTransformer(input_dim, hidden_dim, num_heads, num_layers)
        self.transformer_decoder = TransformerDecoder(input_dim, hidden_dim, num_heads, num_layers, output_dim)
        self.temporal_decoder = TemporalDecoder(input_dim, hidden_dim, num_heads, num_layers, output_dim)
        
    def forward(self, M_sequence):
        # M_sequence is a list of input tensors [M1*, M2*, ..., Mt*] with shape [batch_size, 2HW, C]
        G_sequence = []
        for M in M_sequence:
            attention_output = self.self_attention_transformer(M)
            global_feature = self.transformer_decoder(attention_output)
            G_sequence.append(global_feature)
        
        # Concatenate global features along the temporal dimension
        G_t = torch.stack(G_sequence, dim=1)  # [batch_size, seq_len, 2HW, C]

        # Reshape to [batch_size * 2HW, seq_len, C] for temporal decoding
        G_t = G_t.view(G_t.size(0) * G_t.size(2), G_t.size(1), G_t.size(3))  # [batch_size * 2HW, seq_len, C]

        # Temporal decoding
        integrated_features = self.temporal_decoder(G_t)
        
        # Reshape back to [batch_size, 2HW, seq_len, C]
        integrated_features = integrated_features.view(M_sequence[0].size(0), 2*H*W, sequence_length, C)
        
        # Permute to [batch_size, seq_len, 2HW, C]
        integrated_features = integrated_features.permute(0, 2, 1, 3)
        
        # Split the integrated features into left and right hand
        #G_star_R = integrated_features[:, :, :H*W, :]
        #G_star_L = integrated_features[:, :, H*W:, :]

        return integrated_features

# Hyperparameters
H = 16
W = 12
C = 1280
input_dim = C
hidden_dim = 512
num_heads = 8
num_layers = 6
output_dim = C

# Create an instance of the SIR module
sir = SIR(input_dim, hidden_dim, num_heads, num_layers, output_dim)

# Example input tensor sequence M*
batch_size = 4
sequence_length = 5
M_sequence = [torch.randn(batch_size, 2*H*W, C) for _ in range(sequence_length)]

# Forward pass through the SIR module
# G_star_R, G_star_L = sir(M_sequence)
# print(G_star_R.shape)  # Expected output shape: [batch_size, seq_len, H*W, C]
# print(G_star_L.shape)  # Expected output shape: [batch_size, seq_len, H*W, C]
sir_token = sir(M_sequence)
print(sir_token.shape)

