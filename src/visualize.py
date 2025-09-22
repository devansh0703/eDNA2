import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_abundance(abundance_file):
    df = pd.read_csv(abundance_file)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Cluster', y='Abundance', data=df)
    plt.title('Abundance per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Reads')
    plt.savefig('results/abundance_plot.png')
    plt.show()

if __name__ == "__main__":
    plot_abundance('results/abundance.csv')