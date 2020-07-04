import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import csv

if __name__ == '__main__':
    dataFrame = pd.read_csv('C:/Users/admin/Desktop/train.csv')
    print(dataFrame.head())

    plt.title('Some shit on Histo')
    sns.distplot(dataFrame['landmark_id'])
    plt.show()

    sns.set()
    plt.title('Training set: number of images per class(line plot)')
    landmarks_fold = pd.DataFrame(dataFrame['landmark_id'].value_counts())
    landmarks_fold.reset_index(inplace=True)
    landmarks_fold.columns = ['landmark_id', 'count']
    ax = landmarks_fold['count'].plot(logy=True, grid=True)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=30)
    ax.set(xlabel="Landmarks", ylabel="Number of images")
    plt.show()

    sns.set()
    landmarks_fold_sorted = pd.DataFrame(dataFrame['landmark_id'].value_counts())
    landmarks_fold_sorted.reset_index(inplace=True)
    landmarks_fold_sorted.columns = ['landmark_id', 'count']
    landmarks_fold_sorted = landmarks_fold_sorted.sort_values('landmark_id')
    ax = landmarks_fold_sorted.plot.scatter( \
        x='landmark_id', y='count',
        title='Training set: number of images per class(statter plot)')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=30)
    ax.set(xlabel="Landmarks", ylabel="Number of images")
    plt.show()
