import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_csv("summary/full_summary.csv")
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['eval_top1'] / 100, label='Eval Top1', marker=',')
    plt.plot(df['epoch'], df['eval_top5'] / 100, label='Eval Top5', marker=',')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title('AFT-Simple: Eval Top1 and Top5 Over Epochs')
    plt.show()
