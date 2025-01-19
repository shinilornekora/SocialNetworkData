import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_data():
    data = pd.read_csv('data_with_class_int.csv')

    # Матрица корреляций
    correlation_matrix = data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Матрица корреляций")
    # plt.savefig('correlation_noisy')
    plt.show()
    # Парные графики
    sns.pairplot(data)
    # plt.savefig('fig_noisy.png')
    plt.show()


plot_data()
