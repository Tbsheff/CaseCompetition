# Univariate function
def univariate(df):
  import pandas as pd
  import seaborn as sns
  import matplotlib.pyplot as plt

  df_output = pd.DataFrame(columns=['type', 'missing','unique', 'min', 'q1', 'median', 'q3', 'max', 'mode', 'mean', 'std', 'skew', 'kurt'])


  for col in df:
    # feature that apply to all datasets
    missing = df[col].isna().sum()
    unique = df[col].nunique()
    mode = df[col].mode()[0]

    if pd.api.types.is_numeric_dtype(df[col]):
      #Features for numeric columns only
      mean = round(df[col].mean(), 2)
      min = df[col].min()
      q1 = df[col].quantile(0.25)
      median = df[col].median()
      q3 = df[col].quantile(0.75)
      max = df[col].max()
      std = df[col].std()
      skew = df[col].skew()
      kurt = df[col].kurt()
      df_output.loc[col] = ["numeric", missing, unique, min, q1, median, q3, max, mode, mean, std, skew, kurt]
      sns.histplot(data=df, x=col)
      plt.show()
    else:
      df_output.loc[col] = ["categorical", missing, unique, '-', '-','-','-','-', mode, '-', '-', '-', '-']
      sns.countplot(data=df, x=col)
      plt.show()
  return df_output

def scatterplot(df, feature, label, roundto=4, linecolor='darkorange'):
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns
  from scipy import stats

  #create the plot
  sns.regplot(x=df[feature], y=df[label], line_kws={"color":linecolor})

  # Calculate the regression line
  m, b, r, p, err = stats.linregress(df[feature], df[label])
  tau, tp = stats.kendalltau(df[feature], df[label])
  rho, rp = stats.spearmanr(df[feature], df[label])
  fskew = round(df[feature].skew(), roundto)
  lskew = round(df[label].skew(), roundto)

  # Add all of those values into the plot
  textstr = f'y = {round(m, roundto)}x + {round(b, roundto)}\n'
  textstr += f'r = {round(r, roundto)}, p = {round(p, roundto)}\n'
  textstr += f't = {round(tau, roundto)}, p = {round(tp, roundto)}\n'
  textstr += f'p = {round(rho, roundto)}, p = {round(rp, roundto)}\n'
  textstr += f'{feature} skew = {round(fskew, roundto)}\n'
  textstr += f'{label} skew = {round(lskew, roundto)}'

  plt.text(.95, 0.2, textstr, fontsize=12, transform=plt.gcf().transFigure)
  plt.show()

def bar_chart(df, feature, label, roundto=4, p_threshold=0.05, sig_ttest_only=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats

    # Make sure that the feature is categorical and label is numeric
    if pd.api.types.is_numeric_dtype(df[feature]):
        num = feature
        cat = label
    else:
        num = label
        cat = feature

    # Create the bar chart
    sns.barplot(x=df[cat], y=df[num])

    # Create the numeric lists needed to calculate the ANOVA
    groups = df[cat].unique()
    group_lists = []
    for g in groups:
        group_lists.append(df[df[cat] == g][num])

    f, p = stats.f_oneway(*group_lists) # <-- same as (group_lists[0], group_lists[1], ..., group_lists[n])

    # Calculate individual pairwise t-test for each pair of groups
    ttests = []
    for i1, g1 in enumerate(groups):
        for i2, g2 in enumerate(groups):
            if i2 > i1:
                list1 = df[df[cat]==g1][num]
                list2 = df[df[cat]==g2][num]
                t, tp = stats.ttest_ind(list1, list2)
                ttests.append((f'{g1} - {g2}', round(t, roundto), round(tp, roundto)))

    # Make a Bonferroni correction --> adjust the p-value threshold to be 0.05 / n of ttests
    bonferroni = p_threshold / len(ttests)

    # Create textstr to add statistics to chart
    textstr = f'f: {round(f, roundto)}\n'
    textstr += f'p: {round(p, roundto)}\n'
    textstr += f'Bonferroni p: {round(bonferroni, roundto)}'
    for ttest in ttests:
        if ttest[2] >= bonferroni:
            textstr += f'\n{ttest[0]}: t: {ttest[1]}, p:{ttest[2]}'

    # If there are too many feature groups, print x labels vertically
    if df[feature].nunique() > 7:
        plt.xticks(rotation=90)

    plt.text(.95, 0.10, textstr, fontsize=12, transform=plt.gcf().transFigure)
    plt.show()


def crosstab(df, feature, label, roundto=4):
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import numpy as np

    # Generate the crosstab
    crosstab = pd.crosstab(df[feature], df[label])

    # Calculate the statistics
    X, p, dof, contingency_table = stats.chi2_contingency(crosstab)

    # Display the statistics
    textstr = f'X2: {X}\n'
    textstr += f'p: {p}'
    plt.text(.95, 0.2, textstr, fontsize=12, transform=plt.gcf().transFigure)

    ct_df = pd.DataFrame(np.rint(contingency_table).astype('int64'), columns=crosstab.columns, index=crosstab.index)
    sns.heatmap(ct_df, annot=True, fmt='d', cmap='coolwarm')
    plt.show()

def bivariate(df, label, roundto=4):
    import pandas as pd
    from scipy import stats

    output_df = pd.DataFrame(columns=['missing %', 'skew', 'type', 'unique', 'p', 'r', 'τ', 'ρ', 'y = m(x) + b', 'F', 'X2'])

    for feature in df:
        if feature != label:
          # Calculate stats that apply to all data-types
          df_temp = df[[feature, label]].copy()
          df_temp = df_temp.dropna().copy()
          missing = round((df.shape[0] - df_temp.shape[0]) / df.shape[0], roundto) * 100
          dtype = df_temp[feature].dtype
          unique = df_temp[feature].nunique()
          if pd.api.types.is_numeric_dtype(df[feature]) and pd.api.types.is_numeric_dtype(df[label]):
              # Process N2N relationships
              m, b, r, p, err = stats.linregress(df_temp[feature], df_temp[label])
              tau, tp = stats.kendalltau(df_temp[feature], df_temp[label])
              rho, rp = stats.spearmanr(df_temp[feature], df_temp[label])
              skew = round(df[feature].skew(), roundto)
              output_df.loc[feature] = [f'{missing}%', skew, dtype, unique, round(p, roundto), round(r, roundto), round(tau, roundto),
                                         round(rho, roundto), f"y = {round(m, roundto)}x + {round(b, roundto)}", '-', '-']
              scatterplot(df_temp, feature, label)
          elif not pd.api.types.is_numeric_dtype(df[feature]) and not pd.api.types.is_numeric_dtype(df_temp[label]):
              # Process C2C relationships
              contingency_table = pd.crosstab(df_temp[feature], df_temp[label])
              X2, p, dof, expected = stats.chi2_contingency(contingency_table)
              output_df.loc[feature] = [f'{missing}%', '-', dtype, unique, p, '-', '-', '-', '-', '-', X2]
              crosstab(df_temp, feature, label)
          else:
              # Process C2N and N2C relationships
              if pd.api.types.is_numeric_dtype(df_temp[feature]):
                  skew = round(df[feature].skew(), roundto)
                  num = feature
                  cat = label
              else:
                  skew = '-'
                  num = label
                  cat = feature
                  
              groups = df_temp[cat].unique()
              group_lists = []
              for g in groups:
                  group_lists.append(df_temp[df_temp[cat] == g][num])

              f, p = stats.f_oneway(*group_lists) # <-- same as (group_lists[0], group_lists[1], ..., group_lists[n])
              output_df.loc[feature] = [f'{missing}%', skew, dtype, unique, round(p, roundto), '-', '-', '-', '-', round(f, roundto), '-']
              bar_chart(df_temp, cat, num) 
    
    return output_df.sort_values(by=['p'], ascending=True)

