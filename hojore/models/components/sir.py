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

class SIR(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers, output_dim):
        super(SIR, self).__init__()
        self.self_attention_transformer = SelfAttentionTransformer(input_dim, hidden_dim, num_heads, num_layers)
        self.transformer_decoder = TransformerDecoder(input_dim, hidden_dim, num_heads, num_layers, output_dim)
        self.spatial_decoder = TransformerDecoder(input_dim, hidden_dim, num_heads, num_layers, output_dim)
        self.feature_template = None  # 初始化 feature_template
        
    def forward(self, M):
        # M is a single input tensor with shape [batch_size, 2HW, C]
        if self.feature_template is None:
            self.feature_template = torch.zeros_like(M).mean(dim=1, keepdim=True)  # 初始化 feature_template
        
        attention_output = self.self_attention_transformer(M)
        global_feature = self.transformer_decoder(attention_output)
        
        # 加权更新 global_feature
        global_feature = global_feature * 0.9 + self.feature_template * 0.1
        
        # 通过 spatial_decoder
        integrated_features = self.spatial_decoder(global_feature)
        
        # 更新 feature_template
        self.feature_template = integrated_features * 0.1 + self.feature_template * 0.9
        
        return integrated_features

# 超参数
H = 16
W = 12
C = 1280
input_dim = C
hidden_dim = 512
num_heads = 8
num_layers = 6
output_dim = C

# 创建 SIR 模块实例
sir = SIR(input_dim, hidden_dim, num_heads, num_layers, output_dim)

# 示例输入张量 M
batch_size = 4
M = torch.randn(batch_size, 2*H*W, C)  # 示例输入

# 通过 SIR 模块的前向传递
integrated_features = sir(M)
print(integrated_features.shape)  # 预期输出形状：[batch_size, 2*H*W, C]
