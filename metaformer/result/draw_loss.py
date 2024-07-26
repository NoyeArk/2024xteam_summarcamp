import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    df = pd.read_csv('summary/conv_summary.csv')
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker=',')
    plt.plot(df['epoch'], df['eval_loss'], label='Eval Loss', marker=',')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('AFT-Conv: Training and Evaluation Loss Over Epochs')
    plt.show()
