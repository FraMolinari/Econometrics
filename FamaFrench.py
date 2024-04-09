import datetime as dt
from datetime import date, timedelta
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt



def regression_model (start, end, ticker, alpha):
    alpha=1-alpha
    etf_data = yf.download(ticker, start, end)['Adj Close']
    etf_returns = etf_data.pct_change()

    etf_monthly_returns = etf_returns.resample('M').agg(lambda x: (x + 1).prod() - 1)
    etf_monthly_returns_df = pd.DataFrame({'Return': etf_monthly_returns})

    factors = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start, end)[0]

    etf_monthly_returns_df.index = factors.index
    final_df = pd.merge(etf_monthly_returns_df,factors, on='Date')
    final_df[['Mkt-RF','SMB','HML','RF']] = final_df[['Mkt-RF','SMB','HML','RF']]/100
    final_df['Return-RF'] = final_df.Return - final_df.RF


    y = final_df['Return-RF']
    X = final_df[['Mkt-RF', 'SMB', 'HML']]

    constant = sm.add_constant(X)
    model = sm.OLS(y,constant)
    results= model.fit()

    all_messages = ""

    for factor in X.columns:
        if results.pvalues[factor] < alpha:
            significance = "is significant"
        else:
            significance = "is not significant"
    
        coef = results.params[factor]
        conf_int = results.conf_int(alpha=alpha, cols=None).loc[factor]
        
        message = f"\nWith a significant level of {(1-alpha)*100:.0f}%, the {factor} {significance} in determining the return of the security.\n"
        message += f"For a one-unit increase in the {factor} factor, you would expect the return to increase/decrease by approximately {coef:.4f} units with a confidence interval level of {(1-alpha)*100:.0f}% to increase between {conf_int[0]:.3f} and {conf_int[1]:.3f} units, holding other variables constant.\n"
    
        # Concatenate the message to the all_messages string
        all_messages += message
        all_messages += "\n"

    summary = results.summary()

    return final_df, results, all_messages, summary


def plotting (final_df):
    y = final_df['Return-RF']
    X = final_df[['Mkt-RF', 'SMB', 'HML']]

    constant = sm.add_constant(X)
    model = sm.OLS(y,constant)
    results= model.fit()
    fig, axs = plt.subplots(3, 1, figsize=(5, 15), sharey=True)
    for i, col in enumerate(X.columns):
        sns.scatterplot(x=X[col], y=y, ax=axs[i])

        model = sm.OLS(y, sm.add_constant(X[col])).fit()
        x_range = pd.Series([X[col].min(), X[col].max()])
        axs[i].plot(x_range, model.predict(sm.add_constant(x_range)), color='red', label='Linear Regression')

        axs[i].set_title(f'Scatterplot with Linear Regression Line ({col} vs. Return-RF)')

    
        axs[i].legend()

    fig.text(0.04, 0.5, 'Return-RF', va='center', rotation='vertical')

    plt.tight_layout()
    plotter= plt.show()
    return plotter




def exprected_return(start_2,end_2,ticker_2):
    alpha=5
    linear_regression=regression_model(start_2,end_2, ticker_2, alpha)
    final_df=linear_regression[0]
    risk_free = final_df['RF'].mean()
    market_premium = 0.047/12
    exp_return= risk_free + market_premium*linear_regression[1].params[1]
    final_exp_ret= exp_return*12
    return final_exp_ret






def menu ():
    while True:
        print("Press '1' or '2':\n \n1: Fama French Regression Model\n2: Cost of Equity (Expected Return) Calculator\n")
        choice_1 = input()
        
        if choice_1 == "1":

            factors_for_end_date = web.DataReader('F-F_Research_Data_Factors', 'famafrench', 2023, date.today())[0]
            end_true = factors_for_end_date.index[-1].to_timestamp().date()

            while True:
                start_input = input("Enter the start date (YYYY-MM-DD): ")
                start_year, start_month, start_day = map(int, start_input.split('-'))
                try:
                    start = dt.date(start_year, start_month, start_day)
                except ValueError:
                    print("Please enter a valid date.")
                    continue
                
                end_input = input("Enter the end date (YYYY-MM-DD): ")
                end_year, end_month, end_day = map(int, end_input.split('-'))
                try:
                    end = dt.date(end_year, end_month, end_day)
                except ValueError:
                    print("Please enter a valid date.")
                    continue
                
                ticker = input("Enter the ticker symbol: ")

                if start > end:
                    print("Start Date cannot be after End Date")
                else:
                    break 
                
            if end > end_true:
                end = end_true + timedelta(days=15)

            alpha = float(input("Enter the significance level (e.g., 95 for 95%): ")) / 100

            finale = regression_model(start,end, ticker,alpha)
            print(finale[2])

            while True:
                print("\nPress '1' if you want an overview of the OLS model, Press '2' if you want to see a graphical rapresentation of the Linear Regression, Press '3' to go back to menu :")
                choice_3=input()

                if choice_3 == "1":
                    print(finale[3])
        
                if choice_3 == "2":
                    dataset_plotter = finale[0]
                    plotting(dataset_plotter)

                elif choice_3 == "3":
                    break


        if choice_1 == "2":

            ticker_2 = input("Enter the Ticker symbol: ")
            factors_for_end_date = web.DataReader('F-F_Research_Data_Factors', 'famafrench', 2023, date.today())[0]
            end_true = factors_for_end_date.index[-1].to_timestamp().date()

            end_2 = end_true + timedelta(days=15)
            start_input = input("Enter the number of years used to estimate the beta: ")

            try:
                years = int(start_input)
            except ValueError:
                print("Please enter a valid number of years.")

            start_2 = dt.date(end_2.year - years, end_2.month, end_2.day)

            result=exprected_return(start_2,end_2,ticker_2)  
            print(f"The expected return (cost of equity) of {ticker_2} is {(result*100):.4f}%")  



menu()