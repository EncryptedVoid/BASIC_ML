"""Seaborn is a Python data visualization library based on matplotlib that provides a high-level interface for
drawing attractive and informative statistical graphics. It is particularly suited for visualizing complex datasets
and is built to work well with pandas DataFrames, making it an ideal choice for many data science tasks. Seaborn
simplifies the process of creating certain types of plots, including those that show the relationships between
multiple variables.

The sns.pairplot function is used to create a grid of plots for pairwise relationships in a dataset. The function
creates a scatter plot for each pair of variables in your DataFrame, making it an excellent tool for exploratory data
analysis as it allows you to quickly see correlations, trends, and outliers. Additionally, it can plot a histogram or
a kernel density estimate (KDE) on the diagonal to show the distribution of single variables."""

import seaborn as sns  # Import the seaborn library
import matplotlib.pyplot as plt  # Import matplotlib's pyplot to customize plots further (e.g., display)

from IRIS_DATABASE_SETUP import df

# Use seaborn to visualize the relationships between variables
# Create a pair plot of the DataFrame `df`
# The `hue="target"` parameter colors the points in each plot by the "target" column, which in this case is the species
# of iris.
# This helps to distinguish between different species in each scatter plot.
sns.pairplot(df, hue="target")

# Display the plots
plt.show()
