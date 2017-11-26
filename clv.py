import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifetimes.plotting import plot_frequency_recency_matrix
from tabulate import tabulate


data = pd.read_csv('summary.csv', dtype={'CustomerID': str})
data.set_index('CustomerID', inplace=True)
print(tabulate(data.head(), tablefmt="pipe", headers="keys"))

from lifetimes import BetaGeoFitter

# similar API to scikit-learn and lifelines.
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(data['frequency'], data['recency'], data['T'])
print(bgf)

from lifetimes.plotting import plot_frequency_recency_matrix

plot_frequency_recency_matrix(bgf)
plt.savefig('frequency_recency_matrix.png', dpi=200)
plt.clf()
plt.cla()
plt.close()


from lifetimes.plotting import plot_probability_alive_matrix

plot_probability_alive_matrix(bgf)
plt.savefig('probability_alive_matrix.png', dpi=200)
plt.clf()
plt.cla()
plt.close()


t = 1
data['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    t, data['frequency'], data['recency'], data['T'])
data.sort_values(by='predicted_purchases', ascending=False).head(5)

from lifetimes.plotting import plot_period_transactions
plot_period_transactions(bgf)

plt.savefig('period_transactions.png', dpi=200)
plt.clf()
plt.cla()
plt.close()


transaction_data = pd.read_csv('transaction_data_clean.csv')

from lifetimes.plotting import plot_history_alive
id = 14096
days_since_birth = 200
sp_trans = transaction_data.loc[transaction_data['CustomerID'] == id]
plot_history_alive(bgf, days_since_birth, sp_trans, 'InvoiceDate')
plt.savefig('history_alive.png', dpi=200)
plt.clf()
plt.cla()
plt.close()


returning_customers_summary = data.loc[data['frequency'] > 0]
from lifetimes import GammaGammaFitter

ggf = GammaGammaFitter(penalizer_coef=0)
ggf.fit(returning_customers_summary['frequency'],
        returning_customers_summary['monetary_value'])
print(ggf)

bgf.fit(returning_customers_summary['frequency'],
        returning_customers_summary['recency'], returning_customers_summary['T'])

clv = ggf.customer_lifetime_value(
    bgf,  # the model to use to predict the number of future transactions
    returning_customers_summary['frequency'],
    returning_customers_summary['recency'],
    returning_customers_summary['T'],
    returning_customers_summary['monetary_value'],
    time=12,  # months
    discount_rate=0.01  # monthly discount rate ~ 12.7% annually
)
