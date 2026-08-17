import torch
from torch import nn
from torch.nn import functional
from data_process import device,get_dataloader,my_getdata


# todo 构建编码器解码器
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        """
        :param input_size: 输入seq的词表大小
        :param hidden_size: 隐藏层维度
        :param num_layers: GRU层数
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 输入 [batch_size, seq_len]
        # 输出 [batch_size, seq_len, hidden_size]
        self.embedding = nn.Embedding(input_size, hidden_size)

        # batch_first表示批次是否为第一个维度
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, input, hidden):
        # 数据进入词嵌入层
        # [B, T] -> [B, T, H]
        output = self.embedding(input)

        # 当前输入与当前隐藏状态进入GRU
        # output: [B, T, H]
        # hidden: [num_layers, B, H]
        output, hidden = self.gru(output, hidden)

        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=self.embedding.weight.device
        )


class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1):
        """
        :param output_size: 解码器输出维度，也就是目标词表大小
        :param hidden_size: 解码器隐藏层维度
        :param num_layers: GRU层数
        """
        super().__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)

        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.out = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        # 词嵌入
        # [B, T] -> [B, T, H]
        output = self.embedding(input)

        # relu激活函数
        output = functional.relu(output)

        # GRU层处理
        # output: [B, T, H]
        # hidden: [num_layers, B, H]
        output, hidden = self.gru(output, hidden)

        # [B, T, H] -> [B, T, output_size]
        output = self.out(output)

        # 对目标词表维度进行LogSoftmax
        output = self.softmax(output)

        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=self.embedding.weight.device
        )


class DecoderRNN_withAttention(nn.Module):
    def __init__(self, output_size, hidden_size, dropout_p=0.1, max_length=10):
        """
        带注意力机制的解码器

        Args:
            output_size (int): 目标语言词表大小
            hidden_size (int): GRU 隐藏层维度
            dropout_p (float): dropout 概率
            max_length (int): 编码器输出的最大序列长度（所有句子需 padding 到该长度）
        """
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 词嵌入层：将目标语言单词索引映射为 dense 向量
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # GRU 层：batch_first=True 意味着输入/输出形状为 (B, T, H)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)

        # 输出线性层：将 GRU 输出映射到目标词表大小
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # 注意力分数计算层：输入为 [当前词嵌入 + 上一时刻隐状态] 的拼接（2H 维），
        # 输出长度为 max_length 的分数向量，用于对编码器每个时间步加权
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        # 注意力融合层：将当前词嵌入与注意力上下文向量拼接（2H 维），
        # 映射回 H 维，作为 GRU 的输入
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # Dropout 层
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, input, hidden, encoder_outputs):
        """
        单步前向传播（解码器的一步）

        Args:
            input (torch.LongTensor): 当前时刻的目标语言输入，形状 [batch_size, 1]
            hidden (torch.Tensor): 上一时刻的隐藏状态，形状 [num_layers, batch_size, hidden_size]
            encoder_outputs (torch.Tensor): 编码器所有时刻的输出，形状 [batch_size, seq_len, hidden_size]
                                           注意 seq_len 必须等于 self.max_length

        Returns:
            output (torch.Tensor): 当前时刻的输出概率（log 形式），形状 [batch_size, output_size]
            hidden (torch.Tensor): 更新后的隐藏状态，形状 [num_layers, batch_size, hidden_size]
            attn_weights (torch.Tensor): 注意力权重，形状 [batch_size, max_length]
        """

        # ---------- 1. 词嵌入 ----------
        # input: [B, 1] → embedding → [B, 1, H]
        embedded = self.embedding(input)  # [B, 1, H]

        # 去掉序列维度（因为当前只处理一个时间步，序列长度为 1）
        embedded = torch.squeeze(embedded, 1)  # [B, H]

        # 取出当前时刻的隐藏状态（假设 num_layers=1，所以 hidden 为 [1, B, H]）
        # hidden_top 用于注意力计算，hidden 本身保留给 GRU
        hidden_top = hidden[0]  # [B, H]

        # Dropout 应用于词嵌入
        embedded = self.dropout(embedded)  # [B, H]

        # ---------- 2. 计算注意力权重 ----------
        # 将当前词嵌入和上一时刻隐状态拼接，得到 [B, 2H]
        # 经过 self.attn 线性层得到 [B, max_length] 分数
        attn_weights = functional.softmax(
            self.attn(torch.cat((embedded, hidden_top), -1)),  # [B, 2H]
            dim=-1  # 沿 max_length 维度做 softmax
        )  # [B, max_length]

        # ---------- 3. 计算上下文向量（注意力加权和） ----------
        # attn_weights: [B, max_length] → unsqueeze(1) → [B, 1, max_length]
        # encoder_outputs: [B, max_length, H]
        # bmm 结果: [B, 1, H]
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(1),  # [B, 1, max_len]
            encoder_outputs  # [B, max_len, H]
        )  # [B, 1, H]

        # 去掉序列维度，得到 [B, H]
        attn_applied = torch.squeeze(attn_applied, 1)  # [B, H]

        # ---------- 4. 融合当前词嵌入与注意力上下文 ----------
        # 拼接两个 [B, H] 得到 [B, 2H]
        output = torch.cat((embedded, attn_applied), 1)  # [B, 2H]

        # 通过融合层映射回 [B, H]
        output = self.attn_combine(output)  # [B, H]

        # 增加序列维度，因为 GRU 需要输入 [B, T, H]，这里 T=1
        output = output.unsqueeze(1)  # [B, 1, H]

        # ---------- 5. 通过 GRU ----------
        # GRU 输入: output [B, 1, H], hidden [num_layers, B, H] (此处 num_layers=1)
        # GRU 输出: output [B, 1, H], hidden [num_layers, B, H]
        output, hidden = self.gru(output, hidden)  # output [B, 1, H], hidden [1, B, H]

        # ---------- 6. 输出层 ----------
        # 去掉序列维度，得到 [B, H]
        output = output.squeeze(1)  # [B, H]

        # 线性层映射到词表大小 [B, output_size]
        output = self.out(output)  # [B, output_size]

        # 应用 log_softmax 便于后续使用 NLLLoss
        output = functional.log_softmax(output, dim=1)  # [B, output_size]

        return output,hidden,attn_weights

    def init_hidden(self):
        """
        初始化隐藏状态（当前仅支持 batch_size=1）

        Returns:
            torch.Tensor: 形状 [1, 1, hidden_size]
        """
        return torch.zeros(1, 1, self.hidden_size).to(device)



if __name__ == '__main__':
    pass



