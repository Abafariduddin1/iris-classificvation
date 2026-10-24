import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

flowers = pd.read_csv(r"C:\Users\pirat\Desktop\Iris Classification\iris.csv")
print(flowers.head())
print(flowers.info())

flowers["species"] = flowers["species"].map({"setosa": 0, "versicolor": 1, "virginica": 2})
print(flowers.head())
print(flowers.tail())
plt.subplot(2, 2, 1)
plt.scatter(flowers["sepal_length"], flowers["species"], s=10, c="blue", marker="o")
plt.xlabel("Sepal Length")
plt.ylabel("Species")
plt.title("Iris Classification")

plt.subplot(2, 2, 2)
plt.scatter(flowers["sepal_width"], flowers["species"], s=10, c="red", marker="o")
plt.xlabel("Sepal Width")
plt.ylabel("Species")
plt.title("Iris Classification")
plt.show()
