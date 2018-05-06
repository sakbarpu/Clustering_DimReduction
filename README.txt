This is how to run the code:

python hw5.py data-embedding.csv k

data-embedding.csv is the file of the format mentioned in the handout. k is the value of k in k means.

The global settings are mentioned in the top portion of the  code. The code can be run in one of the following configuration. Only one of the following variables must be true at a time.

isVisualize = 0
UsualSetting = 1
AnalysisB = 0
AnalysisC = 0
Bonus = 0

isVisualize is for exploratory analysis. UsualSetting for running the code for kmeans for the data and k value provided as command line and finding the three scores WC SSD, SC and NMI. The rest are for analysis.

It is important to note that I have used libraries like sklearn and scipy just for the utility functions like finding pairwise distances. No where in the code I am using sklearn or scipy to find the WC SSD, SC or NMI score directly from the library using a single call. Also, I am not using these libraries for finding k means. 

As mentioned in the homework handout, I use scipy for doing hierarchical clustering and for making the dendrogram.