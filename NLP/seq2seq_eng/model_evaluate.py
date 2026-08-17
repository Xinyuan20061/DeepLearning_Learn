import torch
import random
from network_rnn import EncoderRNN,DecoderRNN_withAttention
from data_process import MAX_LENGTH,device,SOS_token,EOS_token,my_getdata


# todo 1、加载训练好的模型
def load_model(english_word_n,french_word_n,hidden_size=256):
    """
    构建并加载训练时保存的编码器与解码器

    :param english_word_n: 英文词表大小
    :param french_word_n: 法文词表大小
    :param hidden_size: 隐藏层维度
    :return: 编码器对象,带注意力的解码器对象
    """
    #构建编码器
    encoder_rnn=EncoderRNN(english_word_n,hidden_size).to(device)
    #构建带注意力的解码器
    attn_decoder_rnn=DecoderRNN_withAttention(french_word_n,hidden_size).to(device)

    #加载训练时保存的权重
    encoder_rnn.load_state_dict(torch.load("./model/encoder_rnn.pth",map_location=device,weights_only=True))
    attn_decoder_rnn.load_state_dict(torch.load("./model/attn_decoder_rnn.pth",map_location=device,weights_only=True))

    #切换到评估模式，关闭dropout
    encoder_rnn.eval()
    attn_decoder_rnn.eval()

    return encoder_rnn,attn_decoder_rnn


# todo 2、评估单条句子
def evaluate_seq2seq(
        x: torch.Tensor,
        encoder_rnn: EncoderRNN,
        attn_decoder_rnn: DecoderRNN_withAttention
):
    """
    将一条已数值化的英语输入解码成法语（贪心解码，不使用teacher_forcing）

    :param x: 英语输入序列 [1,T]
    :param encoder_rnn: 编码器对象
    :param attn_decoder_rnn: 带注意力的解码器对象
    :return: 解码出的单词索引列表,每一步的注意力权重 [步数,MAX_LENGTH]
    """
    with torch.no_grad():
        #初始化encoder的隐藏状态
        encoder_hidden=encoder_rnn.init_hidden(1)
        #获取encoder的输出和隐藏状态
        encoder_output,encoder_hidden=encoder_rnn(x,encoder_hidden)

        #将encoder的输出进行补全到max_length宽度
        encoder_output_c=torch.zeros(1,MAX_LENGTH,encoder_rnn.hidden_size,device=device)
        for idx in range(x.shape[1]):
            encoder_output_c[0][idx]=encoder_output[0][idx]

        #初始化解码器输入与隐藏状态
        input_y=torch.tensor([[SOS_token]],device=device)
        decoder_hidden=attn_decoder_rnn.init_hidden()

        decode_words=[]
        decode_attentions=torch.zeros(MAX_LENGTH,MAX_LENGTH,device=device)

        for step in range(MAX_LENGTH):
            #单步解码
            output_y,decoder_hidden,attn_weight=attn_decoder_rnn(input_y,decoder_hidden,encoder_output_c)
            #取概率最大的词作为预测
            topv,topi=output_y.topk(1)
            pred=topi.squeeze().item()
            #遇到结束标记则停止
            if pred==EOS_token:
                break
            decode_words.append(pred)
            #记录这一步的注意力权重 [1,MAX_LENGTH] -> [MAX_LENGTH]
            decode_attentions[step]=attn_weight.squeeze(0)
            #把预测值作为下一次输入
            input_y=topi.view(1,-1)

        return decode_words,decode_attentions


if __name__=='__main__':
    #获取数据与词汇表
    my_pairs,english_word2index,english_index2word,english_word_n,french_word2index,french_index2word,french_word_n=my_getdata()

    #加载训练好的模型
    encoder_rnn,attn_decoder_rnn=load_model(english_word_n,french_word_n,hidden_size=256)

    #随机抽取几条数据集里的句子做测试
    random.seed(0)
    test_pairs=random.sample(my_pairs,5)

    print("="*50)
    for english_sentence,french_sentence in test_pairs:
        #1、英语句子数值化
        input_ids=[english_word2index[word] for word in english_sentence.split(' ')]
        input_ids.append(EOS_token)
        x=torch.tensor(input_ids,dtype=torch.long,device=device).view(1,-1)

        #2、模型翻译
        decode_words,decode_attentions=evaluate_seq2seq(x,encoder_rnn,attn_decoder_rnn)

        #3、索引转回法语
        pred_sentence=' '.join([french_index2word[idx] for idx in decode_words])

        #打印英语输入与法语输出
        print(f"英语输入: {english_sentence}")
        print(f"真实法语: {french_sentence}")
        print(f"模型翻译: {pred_sentence}")
        print("-"*50)
