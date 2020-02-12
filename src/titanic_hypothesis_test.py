import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

def plot_hist(df, title, color):
    df.hist(figsize=(12, 5), sharey=True, grid=False, color=color, alpha=0.5)
    plt.suptitle(title, size=18, weight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # importing and cleaning the data
    df = pd.read_csv("data/train.csv")
    df.drop(["Cabin", "Embarked", "Ticket"], axis = 1, inplace = True)
    df.dropna(inplace = True)
    df["Sex"].replace({"male" : 0, "female" : 1}, inplace = True)

    #setting the signifcance level
    alpha = .05 / len(df["PassengerId"])

    # eda
    # plot_hist(df, "Titanic Data", "g")

    # seperating male from female
    df_male = df[df["Sex"] == 0].copy()
    df_female = df[df["Sex"] == 1].copy()
    p = df["Survived"].sum() / len(df)

    # binomal test
    p_val = stats.binom_test(df_male['Survived'].sum(), n = len(df_male['Survived']), p = p, alternative = 'less')

    if p_val <= alpha:
        print("Gender was an important factor in surviving the titanic")
    else:
        print("Gender was not an important factor in surviving the titanic")

    fig = plt.figure()
    df_male["Survived"].hist(normed=True, label="Male", color='g', alpha=0.3)
    df_female["Survived"].hist(normed=True, label="Female", color='r', alpha=0.3)
    plt.ylabel("Probabliity Density")
    plt.xlabel('Survived')
    plt.legend()
    plt.show()