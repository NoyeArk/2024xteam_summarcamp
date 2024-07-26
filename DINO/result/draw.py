import matplotlib.pyplot as plt

# 逐行读取每个epoch的输出日志，并把str转成字典
with open('summary/conv_result.txt', 'r') as file:
    dict_list = []
    lines = file.readlines()
    for line in lines:
        line = eval(line)
        dict_list.append(line)

list_train_error = []
list_test_error = []
list_mAP = []
epochs = []
for i in range(12):
    list_train_error.append(dict_list[i]["train_loss"])
    list_test_error.append(dict_list[i]["test_loss"])
    list_mAP.append(dict_list[i]["test_coco_eval_bbox"][0])
    epochs.append(i + 1)

# 可视化
plt.figure(figsize=(10, 6))
# plt.plot(epochs, list_train_error, label='Train Loss', marker=',')
# plt.plot(epochs, list_test_error, label='Train Loss', marker=',')
plt.plot(epochs, list_mAP)
plt.xticks(epochs)
plt.title('AFT-Conv: Eval Map Over Epochs', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Map', fontsize=12)
plt.show(block=False)
