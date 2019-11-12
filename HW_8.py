import csv	
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

wine_file = pd.read_csv('wine.data')
wine_file.head()

def show_correlation(wine_file, figfile=None):
    corr = wine_file.corr()
    fig, ax = plt.subplots()
    im = ax.imshow(corr.values)

    ax.set_title("Wine dataset", fontdict=None, loc='center', pad=None)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)

    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")


    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            text = ax.text(j, i, np.around(corr.iloc[i, j], decimals=2),
                           ha="center", va="center", color="black")

    if figfile:
        plt.savefig(figfile)
    else:
        plt.show()


def show_facet(wine_file, figfile=None):
    wine_file['flavs_rounded'] = wine_file["Flavanoids"].round(1)
    g = sns.FacetGrid(wine_file, hue='flavs_rounded', col='Winery', palette="Blues")
    g = g.map(plt.scatter, 'Nonflavanoid phenols', 'Total phenols', edgecolor='w').add_legend()

    if figfile:
        plt.savefig(figfile)
    else:
        plt.show()
    

def show_density(wine_file, figfile=None):

    fig, axs = plt.subplots(2)
    
    sns.distplot(wine_file['Color intensity'], ax=axs[0])
    sns.scatterplot(x='Color intensity', y='Alcohol', data=wine_file, ax=axs[1], palette='navy')

    if figfile:
        plt.savefig(figfile)
    else:
        plt.show()

show_correlation(wine_file, 'correlation.png')
show_facet(wine_file, 'facet.png')
show_density(wine_file, 'density.png')
