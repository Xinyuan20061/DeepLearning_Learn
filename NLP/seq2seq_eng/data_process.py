import re
import torch
from torch.utils.data import DataLoader,Dataset

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#起始标记
SOS_token=0
#结束标记
EOS_token=1
#最大句子长度
MAX_LENGTH=10
#文件路径
DATA_PATH="./eng-fra-v2.txt"


# todo 2、数据清洗
def normalizeString(s:str):
    """
    :param s: 需要处理的字符串
    :return: 处理后的字符串
    """
    #小写并去除尾部空白
    s=s.lower().strip()
    #正则表达式替换
    s=re.sub('([.!?])',r' \1',s)

    #除了大小写字母和.!?全部过滤
    s=re.sub(r'[^a-zA-Z.!?]+',' ',s)

    return s

# todo 3、数据预处理
def my_getdata():
    with (open(DATA_PATH,'r',encoding='utf-8')as src_f):
        #一次读取所有行
        lines=src_f.readlines()
        #将双语读取为一个列表中的两个字符串
        my_pairs=[ [normalizeString(s) for s in line.split('\t')] for line in lines]

        """
        print(my_pairs[:5])
        [['i m .', 'j ai ans .'], 
        ['i m ok .', 'je vais bien .'], 
        ['i m ok .', 'ca va .'], 
        ['i m fat .', 'je suis gras .'], 
        ['i m fat .', 'je suis gros .']]
        """

        #初始化英语词汇表
        english_word2index={'SOS':SOS_token,'EOS':EOS_token}
        #词汇表大小初始为2：EOS与SOS
        english_word_n=2

        # 初始化法语词汇表
        french_word2index = {'SOS': SOS_token, 'EOS': EOS_token}
        # 词汇表大小初始为2：EOS与SOS
        french_word_n = 2

        for pair in my_pairs:
            for word in pair[0].split(' '):
                #如果单词不在词汇表中，则加入词汇表并分配新索引
                if word not in english_word2index:
                    #分配索引
                    english_word2index[word]=english_word_n
                    #索引自增
                    english_word_n+=1

            for word in pair[1].split(' '):
                # 如果单词不在词汇表中，则加入词汇表并分配新索引
                if word not in french_word2index:
                    # 分配索引
                    french_word2index[word] = french_word_n
                    # 索引自增
                    french_word_n+=1

        #构建反向映射表

        english_index2word={v:k for k,v in english_word2index.items()}
        french_index2word={v:k for k,v in french_word2index.items()}

        return my_pairs,english_word2index,english_index2word,english_word_n,french_word2index,french_index2word,french_word_n

# todo 4、构建数据集对象
class MyPairsDataset(Dataset):
    def __init__(self,my_pairs,english_word2index,french_word2index):
        self.my_pairs=my_pairs
        self.sample_len=len(my_pairs)
        self.english_word2index=english_word2index
        self.french_word2index=french_word2index

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):
        #确保索引在有效范围内
        index=min(max(index,0),self.sample_len-1)

        x=self.my_pairs[index][0]
        y=self.my_pairs[index][1]

        #样本x文本数值化,将x中的每个单词映射为单词表中的索引
        x=[self.english_word2index[word] for word in x.split(' ')]
        # 句子末尾加入结束token
        x.append(EOS_token)
        tensor_x=torch.tensor(x,dtype=torch.long,device=device)

        #样本y文本数值化
        y = [self.french_word2index[word] for word in y.split(' ')]
        # 句子末尾加入结束token
        y.append(EOS_token)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)

        return tensor_x,tensor_y

# todo 5、构建数据加载器对象
def get_dataloader(my_pairs,english_word2index,french_word2index):
    #构建数据集对象
    my_dataset=MyPairsDataset(my_pairs,english_word2index,french_word2index)
    #构建加载器对象
    my_dataloader=DataLoader(dataset=my_dataset,batch_size=1,shuffle=True)

    return my_dataloader


