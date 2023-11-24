""" Create a box plot of the data in a csv file. """

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv('Results/Data_Norm_by_model.csv')

    # Remove any rows that contain all NaN values
    df = df.dropna(how='all')

    # Populate NaN values in df with value in the same column that share the same value in the 'Run Name' column
    df = df.fillna(df.groupby('Run Name').transform('first'))

    print(df)

    # Create 4 box plots, one for metric (SSIM, PSNR, SAM, ERGAS) grouped by 'Data Normalization' (do not share y-axis)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey=False)
    df.boxplot(column=['SSIM'], by='Data Normalization', ax=axes[0,0], rot=45)
    df.boxplot(column=['PSNR'], by='Data Normalization', ax=axes[0,1], rot=45)
    df.boxplot(column=['SAM'], by='Data Normalization', ax=axes[1,0], rot=45)
    df.boxplot(column=['ERGAS'], by='Data Normalization', ax=axes[1,1], rot=45)

    # Remove x-axis labels from top row
    axes[0, 0].set_xticklabels([])
    axes[0, 1].set_xticklabels([])

    # Remove 'Data Normalization' label from x-axis
    axes[0, 0].set_xlabel('')
    axes[0, 1].set_xlabel('')
    axes[1, 0].set_xlabel('')
    axes[1, 1].set_xlabel('')

    fig.suptitle('Model Performance by Data Normalization Method')

    # Display the plot
    plt.show()

