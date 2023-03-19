import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)



df = pd.read_csv('D:\Code_projects\data_analysis\co2_emissions_kt_by_country.csv')

regions_countries = ['Africa Eastern and Southern',
                     'Africa Western and Central',
                     'Arab World',
                    'Central Europe and the Baltics',
                    'East Asia & Pacific',
                    'European Union',
                    'Europe & Central Asia',
                    'Heavily indebted poor countries (HIPC)',
                     'High income',
                     'Low income',
                     'Middle income',
                    'IBRD only',
                     'IDA & IBRD total',
                    'Latin America & Caribbean',
                    'Middle East & North Africa',
                    'Sub-Saharan Africa',
                    'South Asia']


#Visualizing each regions maximum recorded emission levels

filtered_df = df[df['country_name'].isin(regions_countries)]

max_emission = filtered_df.groupby('country_name')['value'].max()

plt.figure(figsize=(15, 60))
plt.pie(max_emission, labels=max_emission.index)
plt.show()

#Plotting the CO2 emission from 1960 to 2019 based on income

income_type = ['Low income', 'Middle income', 'High income']

income_df = df[df['country_name'].isin(income_type)]

plt.figure(figsize=(10, 5))
sns.lineplot(x='year', y='value', hue='country_name', data=income_df)
plt.show()


plt.figure(figsize=(10, 5))

sns.lineplot(x='year', y='value', data=df[df['country_name']=='Romania'])
plt.show()


def forecast(df, country, p, d, q, t):
    '''
    ARIMA model with order(p, d, q)
    p -> lag order
    d -> differencing
    q -> moving average
    Forecast the country's emission for next 't' years
    '''
    
    X_value = df.loc[df['country_name']==country, 'value'].values
    X_year  = df.loc[df['country_name']==country, 'year'].values
    history = [x for x in X_value]
    year = [x for x in X_year]
    test = []
    test_year = []
    
    for x in range(t):
        test_year.append(2020+x)
        model = ARIMA(history, order=(p, d, q))
        fitted_model = model.fit()
        output = fitted_model.forecast()
        test.append(output[0])
        history.append(output[0])
    
    
    plt.figure(figsize=(10, 5))
    plt.plot(year, X_value, label='Emission till 2019')
    plt.plot(test_year, test, label='Emission after 2019')
    plt.legend()
    plt.show()



#global emissions for next 30 years
#this will take some time 
forecast(df, 'World', 15, 2, 5, 30)

#Romania emission for next 30 years
#this will take some time

forecast(df, 'Romania', 15, 2, 5, 30)