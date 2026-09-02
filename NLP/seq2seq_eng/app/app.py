import os
import sys

# 项目根目录（data_process.py / network_rnn.py / model_evaluate.py 所在）
ROOT_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0,ROOT_DIR)

# 资源目录：打包成exe后资源被解压到 sys._MEIPASS，未打包时在项目根目录
if hasattr(sys,'_MEIPASS'):
    RES_DIR=sys._MEIPASS
else:
    RES_DIR=ROOT_DIR

import torch
import tkinter as tk

import data_process
from data_process import device,SOS_token,EOS_token,my_getdata,normalizeString
from network_rnn import EncoderRNN,DecoderRNN_withAttention
from model_evaluate import evaluate_seq2seq


# todo 1、加载数据与模型
def load_all(hidden_size=256):
    """
    加载词汇表与训练好的模型

    :param hidden_size: 隐藏层维度
    :return: 编码器,解码器,英文词汇表,法文反向词汇表
    """
    # 覆盖 data_process 里写死的相对数据路径（打包后 CWD 会变）
    data_process.DATA_PATH=os.path.join(RES_DIR,'eng-fra-v2.txt')

    # 构建词汇表
    my_pairs,english_word2index,english_index2word,english_word_n,french_word2index,french_index2word,french_word_n=my_getdata()

    # 构建模型
    encoder_rnn=EncoderRNN(english_word_n,hidden_size).to(device)
    attn_decoder_rnn=DecoderRNN_withAttention(french_word_n,hidden_size).to(device)

    # 加载训练时保存的权重
    encoder_rnn.load_state_dict(torch.load(os.path.join(RES_DIR,'model','encoder_rnn.pth'),map_location=device,weights_only=True))
    attn_decoder_rnn.load_state_dict(torch.load(os.path.join(RES_DIR,'model','attn_decoder_rnn.pth'),map_location=device,weights_only=True))

    # 切换到评估模式，关闭dropout
    encoder_rnn.eval()
    attn_decoder_rnn.eval()

    return encoder_rnn,attn_decoder_rnn,english_word2index,french_index2word


# todo 2、翻译函数（英语 -> 法语）
def translate(english_sentence,encoder_rnn,attn_decoder_rnn,english_word2index,french_index2word):
    """
    将一句英语翻译成法语

    :param english_sentence: 英语句子（字符串）
    :return: 法语句子（字符串）
    """
    # 清洗英语句子
    english_sentence=normalizeString(english_sentence)
    if english_sentence=='':
        return '请输入英语句子'

    # 分词并数值化
    input_ids=[]
    for word in english_sentence.split(' '):
        # 词表里没有的词直接提示
        if word not in english_word2index:
            return f'未知单词: {word}'
        input_ids.append(english_word2index[word])
    # 句子末尾加入结束标记
    input_ids.append(EOS_token)
    x=torch.tensor(input_ids,dtype=torch.long,device=device).view(1,-1)

    # 贪心解码
    decode_words,_=evaluate_seq2seq(x,encoder_rnn,attn_decoder_rnn)

    # 索引转回法语句子
    french_sentence=' '.join([french_index2word[idx] for idx in decode_words])
    return french_sentence


# todo 3、图形界面
def main():
    # 加载数据与模型（首次启动需要几秒）
    encoder_rnn,attn_decoder_rnn,english_word2index,french_index2word=load_all()

    root=tk.Tk()
    root.title("英语翻译法语")
    root.geometry("480x260")

    # 英语输入框
    tk.Label(root,text="英语输入:").pack(anchor="w",padx=10,pady=(10,0))
    input_entry=tk.Entry(root,width=60)
    input_entry.pack(padx=10,pady=5)

    # 翻译按钮
    def on_translate():
        english_sentence=input_entry.get()
        french_sentence=translate(english_sentence,encoder_rnn,attn_decoder_rnn,english_word2index,french_index2word)
        output_var.set(french_sentence)

    tk.Button(root,text="翻译",command=on_translate).pack(pady=5)

    # 法语输出
    tk.Label(root,text="法语输出:").pack(anchor="w",padx=10,pady=(10,0))
    output_var=tk.StringVar()
    output_var.set("（翻译结果会显示在这里）")
    tk.Label(root,textvariable=output_var,wraplength=440,justify="left").pack(padx=10,pady=5)

    # 回车键也能触发翻译
    input_entry.bind("<Return>",lambda e: on_translate())

    root.mainloop()


if __name__=='__main__':
    main()
