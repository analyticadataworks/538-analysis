import pandas as pd
import statsmodels.api as sm

filepath = 'data/data-20240718.csv'
data = pd.read_csv(filepath)

x = pd.DataFrame({
    "fundonly": data['FundOnly'], 
    "adjpollsonly": data['AdjPollsOnly']
})
y = data['Final']
model = sm.OLS(exog=sm.add_constant(x), endog=y)
result = model.fit()

print(result.summary())