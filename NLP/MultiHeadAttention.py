import torch
from torch import nn

class DotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.5)

    def forward(
            self,
            q:torch.Tensor,
            k:torch.Tensor,
            v:torch.Tensor
    ):
        """
        :param q:[batch_size, seq_q, d_k]
        :param k:[batch_size, seq_k, d_k]
        :param v:[batch_size, seq_k, d_v]
        :return:
        """
        d=q.size(-1)  # d = d_k，Q/K的最后一维维度
        scores=torch.bmm(q,k.transpose(1,2)) / d ** 0.5  # scores: [batch_size, seq_q, seq_k]
        attention_weights=nn.functional.softmax(scores,dim=-1)  # attention_weights: [batch_size, seq_q, seq_k]

        return torch.bmm(self.dropout(attention_weights),v)  # output: [batch_size, seq_q, d_v]


class MultiHeadAttention(nn.Module):
    def __init__(self,key_size,query_size,value_size,num_hidden,num_head):
        """
        :param key_size: K原始特征维度
        :param query_size: Q原始特征维度
        :param value_size: V原始特征维度
        :param num_hidden: 多头拼接后的总隐藏维度，num_hidden = num_head * d_head
        :param num_head: 头的数量
        """
        super().__init__()
        self.num_head=num_head
        self.attention=DotProductAttention()
        self.W_q=nn.Linear(query_size,num_hidden)
        self.W_k=nn.Linear(key_size,num_hidden)
        self.W_v=nn.Linear(value_size,num_hidden)
        self.W_o=nn.Linear(num_hidden,value_size)

    def forward(self,Q,K,V):
        """
        :param Q:[batch_size, seq_q, d_k]
        :param K:[batch_size, seq_k, d_k]
        :param V:[batch_size, seq_k, d_v]
        :return:
        """

        #先让Q,K,V通过线性层
        Q=self.W_q(Q)
        K=self.W_k(K)
        V=self.W_v(V)

        #切分注意力实现多头注意力机制
        Q=self.split_heads(Q)
        K=self.split_heads(K)
        V=self.split_heads(V)

        #进行注意力计算并返回
        output=self.attention(Q,K,V)
        output=self.transpose_heads(output)
        return self.W_o(output)



    #切分注意力实现多头注意力机制
    def split_heads(self,x:torch.Tensor):
        """形状变化
        1、[batch_size, seq, d] -> [batch_size, seq, num_head, d_k]
        2、[batch_size, seq, num_head, d_k] -> [batch_size, num_head, seq, d_k]
        3、[batch_size, num_head, seq, d_k] -> [batch_size*num_head, seq, d_k]
        """
        x=x.reshape(x.shape[0],x.shape[1],self.num_head,-1)
        x=x.permute(0,2,1,3)
        return x.reshape(-1,x.shape[2],x.shape[3])

    def transpose_heads(self,x:torch.Tensor):
        """形状变化
        1、[batch_size*num_head, seq, d_k] -> [batch_size, num_head, seq, d_k]
        2、[batch_size, num_head, seq, d_k] -> [batch_size, seq, num_head, d_k]
        3、[batch_size, seq, num_head, d_k] -> [batch_size, seq, d_v]
        """
        x=x.reshape(-1,self.num_head,x.shape[1],x.shape[2])
        x=x.permute(0,2,1,3)
        return x.reshape(x.shape[0],x.shape[1],-1)

if __name__=='__main__':
    print(MultiHeadAttention(10,10,10,10,10))
