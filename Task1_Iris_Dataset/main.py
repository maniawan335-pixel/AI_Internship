import pandas as pd
import numpy as numpy
import seaborn as sns
import matplotlib.pyplot as plt

# we are going to use the iris data set from seaborn, lets load it.

df = sns.load_dataset("iris")

# lets display the first 5 rows of the data set
print(df.head())

# lets check out the shape of the dataset.

print(df.shape)

# lets check the columns of the dataset

print(df.columns)

# lets check the information of the dataaset

print(df.info())

# lets check the description of the dataaset

print(df.describe())

# lets draw the scatter plot

sns.scatterplot(
    x="sepal_length",
    y="sepal_width",
    hue="species",
    data=df
)
plt.show()
sns.scatterplot(
    x="petal_length",
    y="petal_width",
    hue="species",
    data=df
)

plt.show()

# lets draw the histogram

sns.histplot(
    x="sepal_length",
    hue="species",
    data=df
)
plt.show()

sns.histplot(
    x="sepal_width",
    hue="species",
    data=df
)
plt.show()

sns.histplot(
    x="petal_length",
    hue="species",
    data=df
)
plt.show()

sns.histplot(
    x="petal_width",
    hue="species",
    data=df
)
plt.show()

sns.boxplot(
    x="sepal_length",
    y="species",
    data=df
)
plt.show()

sns.boxplot(
    x="sepal_width",
    y="species",
    data=df
)
plt.show()

sns.boxplot(
    x="petal_length",
    y="species",
    data=df
)
plt.show()

sns.boxplot(
    x="petal_width",
    y="species",
    data=df
)
plt.show()