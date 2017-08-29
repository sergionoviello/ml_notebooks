# Machine Learning cookbook
This cookbook contains snippet of code that I use for my machine learning projects.

### Helpers

```
def describe_categorical(X):
    print(X[X.columns[X.dtypes == 'object']].describe())
    
def cstats(y_test, y_test_pred):
    return roc_auc_score(y_test, y_test_pred)

def get_original_datasets(idx):
    global combined
    
    train0 = pd.read_csv('train.csv')
    
    targets = train0.Survived
    train = combined.head(idx)
    test = combined.iloc[idx:]
    
    return train, test, targets

def combined_dataset():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)
    
    return combined, train.shape[0]

```


## Data Preprocessing

##### Display dtypes of features
```
types = train.columns.to_series().groupby(train.dtypes).groups
for k,v in types.items():
    print(k, v)
```

##### Display feautures null values

```
train.isnull().sum()
```


## Exploratory Data Analysis

#### Distribution of a numerical variable (age)

```
median = train['Age'].median()
sns.distplot(train['Age'].dropna(), bins=30)
plt.plot([median, median], [0, 0.05], linewidth=2, c='r')
```

#### Describe Categorical features
```
def describe_categorical(X):
    print(X[X.columns[X.dtypes == 'object']].describe())
```

#### Graph most important features
```
def feature_importances(model, feature_names, autoscale=True, margin=0.05, sum_cols=None, width=5):
    if autoscale:
        x_scale = model.feature_importances_.max() + margin
    else:
        x_scale = 1

    feature_d = dict(zip(feature_names, model.feature_importances_))

    if sum_cols:
        for col in sum_cols:
            val = sum(x for i, x in feature_d.items() if col in i)
            keys_to_remove = [i for i in feature_d.keys() if col in i]
            for i in keys_to_remove:
                feature_d.pop(i)
            feature_d[col] = val
    results = pd.DataFrame.from_dict(feature_d, orient='index')
    results.columns = ['cat']

    results.sort_values('cat', ascending=True, inplace=True)
    results.plot(kind='barh', figsize=(width, len(results)/2), xlim=(0, x_scale), legend=None)

feature_importances(model, train.columns, sum_cols=categorical_variables)

```

## Feature engineering

#### combine test and train
```
def combined_dataset():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    targets = train.Survived
    train.drop('Survived', 1, inplace=True)
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index', inplace=True, axis=1)

    return combined
```

then create a function for each feature I want to change or create

```
def process_cabin():

    global combined
    combined.Feature.fillna('U', inplace=True)
    combined['Feature'] = combined['Feature'].map(lambda c : c[0])

    dummies = pd.get_dummies(combined['Feature'], prefix='Feature')
    combined = pd.concat([combined,dummies], axis=1)
    combined.drop('Feaure', axis=1, inplace=True)

 ```

 or just to fill null values
 ```
 def process_feature():

    global combined
    combined.head(891).Feature.fillna(combined.head(891).Feature.mean(), inplace=True)
    combined.iloc[891:].Feature.fillna(combined.iloc[891:].Feature.mean(), inplace=True)
 ```


### Evaluation

##### Classification

```
def cstats(y_test, y_test_pred):
    return roc_auc_score(y_test, y_test_pred)
    
print('training set:', cstats(y_train, y_train_pred))
print('validation set:', cstats(y_test, y_test_pred))
```

```
kfold = KFold(n_splits=10, random_state=7)
scores = cross_val_score(model, X_train, y_train, cv=kfold)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```

```
print(classification_report(y_test, y_test_pred))
```

# References

#### metrics
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
