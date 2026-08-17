import time
import torch
from torch.optim import Adam
import random
from network_rnn import EncoderRNN,DecoderRNN_withAttention
from data_process import MAX_LENGTH,device,SOS_token,EOS_token,get_dataloader,my_getdata
from matplotlib import pyplot as plt

epochs=100
lr=1e-4
teacher_forcing_ratio=0.5
print_interval_num=100
plot_interval_num=100

# todo 单批次训练函数
#内部迭代训练函数
def Train_Iters(
        x :torch.Tensor,
        y :torch.Tensor,
        encoder_rnn :EncoderRNN,
        attn_decoder_rnn :DecoderRNN_withAttention,
        adam_encode :Adam,
        adam_decode :Adam,
        LossFunction
):
    """
    :param x:输入序列 [batch_size=1,seq_len] 即英语句子
    :param y:目标序列 [batch_size=1,seq_len] 即法语句子
    :param encoder_rnn:编码器对象
    :param attn_decoder_rnn:解码器对象
    :param adam_encode:编码器优化器
    :param adam_decode:解码器优化器
    :param LossFunction:损失函数
    :return:
    """

    #一次推送数据
    #初始化隐藏状态[1,1,hidden_size]
    encoder_hidden=encoder_rnn.init_hidden(batch_size=1)
    encoder_output,encoder_hidden=encoder_rnn(x,encoder_hidden)

    #将output填充为max_length维度的张量
    #编码器输出张量[max_length,hidden_size]
    encoder_output_c=torch.zeros(1,MAX_LENGTH,encoder_rnn.hidden_size,device=device)
    for idx in range(x.shape[1]):
        encoder_output_c[0][idx]=encoder_output[0][idx]

    #初始化解码器hidden输入[1,1,hidden_size]
    decoder_hidden=attn_decoder_rnn.init_hidden()

    #初始化解码器初始输入[1,1]
    input_y=torch.tensor([[SOS_token]],device=device)

    loss=0.0
    y_len=y.shape[1]

    #判断是否使用teacher_forcing
    user_teacher_forcing=True if random.random() <teacher_forcing_ratio else False

    if user_teacher_forcing:
        for idx in range(y_len):
            output_y,decoder_hidden,attm_weight=attn_decoder_rnn(input_y,decoder_hidden,encoder_output_c)
            target_y=y[0][idx].view(1)
            loss+=LossFunction(output_y,target_y)
            #如果使用teacher_forcing，则把样本真实值作为下一次输入
            input_y=y[0][idx].view(1,-1) # [1,1] “一个词”
    else:
        for idx in range(y_len):
            output_y,decoder_hidden,attm_weight=attn_decoder_rnn(input_y,decoder_hidden,encoder_output_c)
            target_y=y[0][idx].view(1)
            loss+=LossFunction(output_y,target_y)
            #如果不用teacher_forcing，则把预测值y作为下一次输入
            topv,topi=output_y.topk(1)
            input_y=topi.view(1,-1)
            #如果检测到结束标记，则停止
            if topi.squeeze().item()==EOS_token:
                break


    #梯度清零+反向传播+参数更新
    adam_encode.zero_grad()
    adam_decode.zero_grad()
    loss.backward()
    adam_encode.step()
    adam_decode.step()

    #返回平均损失值
    return loss.item()/y_len

# todo 训练函数
def Train_seq2seq():
    #获取数据加载器
    my_pairs, english_word2index, english_index2word, english_word_n, french_word2index, french_index2word, french_word_n=my_getdata()
    dataloader=get_dataloader(my_pairs,english_word2index,french_word2index)

    #模型初始化
    encoder_rnn=EncoderRNN(english_word_n,256).to(device)
    attn_decoder_rnn=DecoderRNN_withAttention(french_word_n,256).to(device)

    #优化器初始化
    adam_encode=Adam(encoder_rnn.parameters(),lr=lr)
    adam_decode=Adam(attn_decoder_rnn.parameters(),lr=lr)

    #损失函数初始化
    LossFunction=torch.nn.NLLLoss()

    #绘图参数初始化
    plot_loss_list=[]

    for epoch in range(1,epochs+1):
        #初始化本轮的损失累加
        total_loss,plot_total_loss=0,0

        #记录本轮开始训练时间
        start_time=time.time()

        for item,(x,y) in enumerate(dataloader,start=1):
            #调用内部训练函数
            loss=Train_Iters(x,y,encoder_rnn,attn_decoder_rnn,adam_encode,adam_decode,LossFunction)

            #记录损失
            total_loss+=loss
            plot_total_loss+=loss

            #日志打印
            if item % print_interval_num==0:
                print(f"Epoch [{epoch}/{epochs}], Step [{item}/{len(dataloader)}], Loss: {loss:.4f}")


            if item>100:
                break

        #训练时间长，每轮训练完保存一次模型
        torch.save(encoder_rnn.state_dict(), f"./model/encoder_rnn.pth")
        torch.save(attn_decoder_rnn.state_dict(), f"./model/attn_decoder_rnn.pth")

        #记录本轮损失
        plot_loss_list.append(plot_total_loss)

    #绘图
    plt.figure()
    plt.plot(plot_loss_list)
    plt.savefig('./img/seq2seq_loss.png')
    plt.show()

    return plot_loss_list





if __name__=='__main__':
    Train_seq2seq()



