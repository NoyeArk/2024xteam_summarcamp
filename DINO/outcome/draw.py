import matplotlib.pyplot as plt

# 逐行读取每个epoch的输出日志，并把str转成字典
with open('log.txt', 'r') as file:
    dict_list = []
    lines = file.readlines()
    for line in lines:
        line = eval(line)
        dict_list.append(line)

list_train_error = []
list_test_error = []
list_mAP = []
epochs = []
for i in range(4):
    list_train_error.append(dict_list[i]["train_loss"])
    list_test_error.append(dict_list[i]["test_loss"])
    list_mAP.append(dict_list[i]["test_coco_eval_bbox"][0])
    epochs.append(i + 1)

# 可视化
plt.figure()
plt.plot(epochs, list_mAP)
plt.xticks(epochs)
# plt.yticks(range(min(list_train_error),max(list_train_error)+1))
plt.title('list_mAP vs. Epoch', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('list_mAP', fontsize=12)
plt.show(block=False)