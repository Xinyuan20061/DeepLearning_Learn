import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import string
from tqdm import tqdm


# 常用字符
all_letters = string.ascii_letters + " ,;.'"
# 常用字符数量
n_letters = len(all_letters)

# 国家列表
categorys = ['Czech', 'German', 'Arabic', 'Japanese', 'Chinese', 'Vietnamese', 'Russian', 'French', 'Irish', 'English',
             'Spanish', 'Greek', 'Italian', 'Portuguese', 'Scottish', 'Dutch', 'Korean', 'Polish']

# 路径
file_path = "data/name_classfication.txt"


class net(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden):
        input = input.transpose(0, 1)
        output, hidden = self.rnn(input, hidden)

        output = output[-1]
        output = self.out(output)

        return output, hidden

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        c=hidden
        return (hidden,c)


"""
读取数据到列表中
"""


def readFile(file_path):
    list_x = []  # 人名列表
    list_y = []  # 国家列表

    with open(file_path, encoding="utf-8") as f:
        for line in f:
            # 数据清洗
            if len(line) < 5:
                continue

            x, y = line.strip().split("\t")
            list_x.append(x)
            list_y.append(y)

    return list_x, list_y


"""
自定义dataset类

"""


class MyDataset(Dataset):
    def __init__(self, list_x, list_y):
        super().__init__()

        # 加载数据
        self.list_x = list_x
        self.list_y = list_y

        # 样本数量
        self.sample_len = len(list_x)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):
        tensor_x, tensor_y = self.data_process(index)
        return tensor_x, tensor_y

    def data_process(self, index):
        x = self.list_x[index]
        y = self.list_y[index]

        """
        对样本x one-hot编码化
        每一个字母都变为one-hot编码
        """
        tensor_x = torch.zeros((len(x), n_letters))
        for idx, letter in enumerate(x):
            tensor_x[idx][all_letters.find(letter)] = 1

        tensor_y = torch.tensor(categorys.index(y))

        return tensor_x, tensor_y


def train():
    # 创建数据加载器
    list_x, list_y = readFile(file_path)
    train_dataset = MyDataset(list_x, list_y)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    model = net(n_letters, 128, 1, len(categorys))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20

    for epoch in range(epochs):
        total_loss, batch_num = 0, 0

        for x, y in tqdm(train_loader):
            hidden = model.init_hidden()
            output, hidden = model(x, hidden)

            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_num += 1

        print("epochs:", epoch, "loss:", total_loss / batch_num)

    torch.save(model.state_dict(), "model/nameClassfication.pth")


def evaluate(name):
    # 将name变为one-hot编码
    tensor_x = torch.zeros((len(name), n_letters))
    for idx, letter in enumerate(name):
        tensor_x[idx][all_letters.find(letter)] = 1

    tensor_x=torch.unsqueeze(tensor_x,0)

    print(tensor_x.shape)#torch.Size([7, 1, 57])

    model = net(n_letters, 128, 1, len(categorys))
    model.load_state_dict(torch.load("model/nameClassfication.pth"))


    hidden = model.init_hidden()
    output, hn = model(tensor_x, hidden)

    print(categorys[output.argmax(dim=1).item()])

if __name__ == "__main__":
    evaluate("Chen")
