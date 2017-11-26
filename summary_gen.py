import pandas as pd
import numpy as np
from lifetimes.utils import summary_data_from_transaction_data
from tabulate import tabulate

b_dir = './ecommerce-data/'
transaction_data = pd.read_csv(
    b_dir + 'data.csv', encoding='latin1', dtype={'CustomerID': str, 'InvoiceID': str})

transaction_data = transaction_data.loc[~transaction_data.CustomerID.isnull()]
transaction_data = transaction_data.loc[transaction_data.UnitPrice > 0]
transaction_data = transaction_data.loc[transaction_data.Quantity > 0]

transaction_data['InvoiceDate'] = pd.to_datetime(
    transaction_data['InvoiceDate'])


transaction_data['monetary_value'] = transaction_data['Quantity'] * \
    transaction_data['UnitPrice']

summary = summary_data_from_transaction_data(
    transaction_data, 'CustomerID', 'InvoiceDate', 'monetary_value')


print(tabulate(summary.head(), tablefmt="pipe", headers="keys"))

summary.to_csv('summary.csv')
transaction_data.to_csv('transaction_data_clean.csv')
