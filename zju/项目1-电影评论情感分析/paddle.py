import paddle
import paddle.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# 生成句子列表
def ids_to_str(ids):
    print(ids)
    words = []
    for k in ids:
        w = list(word_dict)[k]
        words.append(w if isinstance(w, str) else w.decode('ASCII'))
    return " ".join(words)


# 读取数据归一化处理
def create_padded_dataset(dataset):
    padded_sents = []
    labels = []
    for batch_id, data in enumerate(dataset):
        sent, label = data[0], data[1]
        padded_sent = np.concatenate([sent[:seq_len], [pad_id] * (seq_len - len(sent))]).astype('int32')
        padded_sents.append(padded_sent)
        labels.append(label)
    return np.array(padded_sents), np.array(labels)


class IMDBDataset(paddle.io.Dataset):
    def __init__(self, sents, labels):
        self.sents = sents
        self.labels = labels

    def __getitem__(self, index):
        data = self.sents[index]
        label = self.labels[index]

        return data, label

    def __len__(self):
        return len(self.sents)


# 定义RNN网络
class RNN(paddle.nn.Layer):
    def __init__(self):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.rnn = nn.SimpleRNN(256, 256, num_layers=2, direction='forward', dropout=0.5)
        self.linear = nn.Linear(in_features=256 * 2, out_features=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        emb = self.dropout(self.embedding(inputs))
        # output形状大小为[batch_size,seq_len,num_directions * hidden_size]
        # hidden形状大小为[num_layers * num_directions, batch_size, hidden_size]
        # 把前向的hidden与后向的hidden合并在一起
        output, hidden = self.rnn(emb)
        hidden = paddle.concat((hidden[-2, :, :], hidden[-1, :, :]), axis=1)
        # hidden形状大小为[batch_size, hidden_size * num_directions]
        hidden = self.dropout(hidden)
        return self.linear(hidden)


def train(model):
    model.train()
    opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    steps = 0
    Iters, total_loss, total_acc = [], [], []

    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader):
            steps += 1
            sent = data[0]
            label = data[1]

            logits = model(sent)
            loss = paddle.nn.functional.cross_entropy(logits, label)
            acc = paddle.metric.accuracy(logits, label)

            if batch_id % 500 == 0:  # 500个epoch输出一次结果
                Iters.append(steps)
                total_loss.append(loss.numpy()[0])
                total_acc.append(acc.numpy()[0])

                print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))

            loss.backward()
            opt.step()
            opt.clear_grad()

        # evaluate model after one epoch
        model.eval()
        accuracies = []
        losses = []

        for batch_id, data in enumerate(test_loader):
            sent = data[0]
            label = data[1]

            logits = model(sent)
            loss = paddle.nn.functional.cross_entropy(logits, label)
            acc = paddle.metric.accuracy(logits, label)

            accuracies.append(acc.numpy())
            losses.append(loss.numpy())

        avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)

        print("[validation] accuracy: {}, loss: {}".format(avg_acc, avg_loss))

        model.train()

        # 保存模型
        paddle.save(model.state_dict(), str(epoch) + "_model_final.pdparams")

    # 可视化查看
    draw_process("trainning loss", "red", Iters, total_loss, "trainning loss")
    draw_process("trainning acc", "green", Iters, total_acc, "trainning acc")


# 可视化定义
def draw_process(title, color, iters, data, label):
    plt.title(title, fontsize=24)
    plt.xlabel("iter", fontsize=20)
    plt.ylabel(label, fontsize=20)
    plt.plot(iters, data, color=color, label=label)
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    train_dataset = paddle.text.datasets.Imdb(mode='train')
    test_dataset = paddle.text.datasets.Imdb(mode='test')

    word_dict = train_dataset.word_idx  # 获取数据集的词表
    word_dict['<pad>'] = len(word_dict)

    for k in list(word_dict)[:5]:
        print("{}:{}".format(k, word_dict[k]))

    print("...")

    for k in list(word_dict)[-5:]:
        print("{}:{}".format(k, word_dict[k]))

    print("一共{}个单词".format(len(word_dict)))

    vocab_size = len(word_dict) + 1
    print(vocab_size)
    emb_size = 256
    seq_len = 200
    batch_size = 32
    epochs = 2
    pad_id = word_dict['<pad>']

    classes = ['negative', 'positive']

    # 对train、test数据进行实例化
    train_sents, train_labels = create_padded_dataset(train_dataset)
    test_sents, test_labels = create_padded_dataset(test_dataset)

    train_dataset = IMDBDataset(train_sents, train_labels)
    test_dataset = IMDBDataset(test_sents, test_labels)

    train_loader = paddle.io.DataLoader(train_dataset, return_list=True,
                                        shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = paddle.io.DataLoader(test_dataset, return_list=True,
                                       shuffle=True, batch_size=batch_size, drop_last=True)

    model = RNN()
    train(model)
