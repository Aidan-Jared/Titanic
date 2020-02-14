# Titanic

## Introduction
This is a practice project to work with h2o.ai and scipy.stats on the titanic dataset from kaggle

## Data
The data set gives a list of the passengers on the titanic and wether or not they survived.

## Hypothesis Test
The first question I wanted to ask was if gender played a significant part in suriving the titanic. In order to answer this question I decided to do a Binomial Hypothesis test because in the case of the titanic you either surved or you did not giving two possible results which Binomials are all about.
First I read in the data using pandas and removed some mostly empty columns then droped all empty values from the data frame. Then I changed the gender of the passenger from male and female to 0 and 1 to make some of the calculations easier.

```python
    df = pd.read_csv("data/train.csv")
    df.drop(["Cabin", "Embarked", "Ticket"], axis = 1, inplace = True)
    df.dropna(inplace = True)
    df["Sex"].replace({"male" : 0, "female" : 1}, inplace = True)
```

Then I chose my alpha to be .05 and adjusted it for the total number of passengers on board the titanic which changed it to 7.0e-5. I also found the probablity of surviving the titanic by taking the total number of people who survived and then deviding by the length of the dataframe.

```python
    alpha = .05 / len(df["PassengerId"])
    p = df["Survived"].sum() / len(df)
```

The last step was to then run the binomial test in the scipy.stats package.

```python
    p_val = stats.binom_test(df_male['Survived'].sum(), n = len(df_male['Survived']), p = p, alternative = 'less')
```

This resulted in a p_val of 7.5e-20 which is less than alpha meaning I could reject the null and show that gender was an important part of surviving the Titanic.

## Machine Learning
