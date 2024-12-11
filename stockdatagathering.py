#Imports:
# requests library allows for HTTP requests for fetching data from APIs
import requests

# beautifulSoup allos HTML parsing and web crawling to get data from web pages
from bs4 import BeautifulSoup

# pipeline allows access to trained NLP models and "sentiment-analysis" is the model we chose
from transformers import pipeline
sentiment = pipeline("sentiment-analysis")

# tokenizer and modelForSequenceClassification allow use of BERTweet for tweet sentiment analysis
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# torch allows for softmax and tensor handling
import torch

# defaultdict allows a dict with a default value for missing keys
from collections import defaultdict

# json library allows json actions like reading in a json file
import json

# datetime library allows for date formatting
from datetime import datetime, timedelta

# LinearRegression model is needed to predict sentiment scores based off historical data
from sklearn.linear_model import LinearRegression

# sklearn.metrics allows for many of the metrics commonly used in linear regression to be imported
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

# train_test_split lets the regression model split data into training data for the model and testing data to run the model on
from sklearn.model_selection import train_test_split

# numpy allows access to number, array, and math functions
import numpy as np

# pandas allows easier use of pyplot for plotting the data
import pandas as pd

# matplotlib.pyplot allows the plotting and display of graphs and charts for the analysis of data
import matplotlib.pyplot as plt

# ticker allow changes to the plots' axis format
import matplotlib.ticker as ticker

#Step 1:ata Collection
# Gathering the Daily Open and Close times of Meta of past 10 years
# use the Alpha Vantage API to get historical stock price data
def get_daily_open_close_prices():
    # the base url of Alpha Ventage's API for historical financial data
    base_URL = "https://www.alphavantage.co/query"

    # the api key - limit = 25 calls a day
    # alternative_api_key = 'VPTDPQ64TZ19TDT7'
    api_key = 'PQRD931A81GZAULM'

    # the API parameters set to find daily stock prices of $META as a full data series
    params ={
        "function": "TIME_SERIES_DAILY",
        "symbol": "META",
        "apikey": api_key,
        "outputsize": "full"
    }

    # set a GET response to Alpha Vantage's API with the parameters
    response = requests.get(base_URL, params=params)

    # convert the API response to JSON for extraction
    meta_data = response.json()

    # initializes a list for the date, open price, and close price
    daily_open_close = []

    # for each day in the time series from 2014 onwards, append a day's date, open price and close price
    for key, value in meta_data['Time Series (Daily)'].items():
      if int(key[:4]) > 2013:
        daily_open_close.append([key, value['1. open'], value['4. close']])
    return daily_open_close

# here we used APIFY, as a third party, to gather twitter post from the past decade about the stock tickers FB and META and access it through "combined.json".
def format_tweets():
    # load in "combined.json" which holds all the twitter data about META and FB
    with open('/content/combined.json', 'r') as file:
        data = json.load(file)

    # extracts the fullText and createdAt elements from each tweet, but may need adjusting based on the nesting in your JSON
    pages = data['data']

    # initializes a list for the tweets' date and text content after formatting
    tweet_list = []

    # loop through all the tweets
    for i in range(len(pages)):
        tweets = pages[i]
        for tweet in tweets:
            full_text = tweet.get('fullText', 'No text found')
            created_at = tweet.get('createdAt', 'No date found')

            # attempts to format the tweet's date into a YYYY-MM-DD format if possible
            try:
                date_object = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y")
                formatted_date = date_object.strftime("%Y-%m-%d")
            except ValueError:
                formatted_date = "Invalid date format"

            # append the tweet's date and text content
            tweet_list.append([formatted_date, full_text])
    return tweet_list

#Step 2: Sentiment Analysis
def analyze_tweet_sentiment(tweet_list):
    # initializes a list for the analyzed sentiment scores
    analysis = []

    # the model type is BERTweet is chosen as an NLP focused on tweet sentiment analysis
    model_name = "finiteautomata/bertweet-base-sentiment-analysis"

    # load the tokenizer for BERTweet
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # load the model for BERTweet
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # the name of the targeted ticker symbol
    target = ""

    # loop through each tweet in tweetList
    for tweet in tweet_list:

        # grab the tweet's text content
        sentence = tweet[1]

        # looks for '$META" or the older ticker symbol '$FB'
        if "$META" in sentence:
            target = "$META"
        else:
            target = "$FB"

        # tokenize the text content with truncation and padding
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)

        # BERTweet model outputs raw predictions as logits
        outputs = model(**inputs)
        logits = outputs.logits

        # convert logits to probabilities with torch's softmax
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        # gets the individual negative, neutral, and positive sentiment probabilities
        sentiment_scores = probabilities.detach().numpy()[0]
        negative, neutral, positive = sentiment_scores

        # calculates the overall sentiment score
        overall_score = (-1 * negative) + (0 * neutral) + (1 * positive)

        # append the analysis to the list with the tweet's date, ticker symbol, and sentiment score
        analysis.append([tweet[0], target, overall_score])
    return analysis

# after performing sentiment analysis we are calculating the daily sentiment score set between -100 and 100.
def calculate_daily_sentiment_scores(analysis):
    # defaultdict can hold counts of the scores classified either positive or negative for each date
    sentiment_dict = defaultdict(lambda: {"pos": 0, "neg": 0})

    # loop through the analyzed list to get the counts of positive and negative scored tweets
    for date, ticket, score in analysis:
        if score >= 0:
            sentiment_dict[date]["pos"] += 1
        elif score < 0:
            sentiment_dict[date]["neg"] += 1

    # does the daily sentiment score calculation based off the number of positive and negative sentiment score posts
    daily_sentiment_scores = []
    for date, count in sentiment_dict.items():
        N_pos = count["pos"]
        N_neg = count["neg"]

        daily_score = ((N_pos - N_neg) / (N_pos + N_neg)) * 100
        daily_sentiment_scores.append({"Date": date, "Score": daily_score})
    return daily_sentiment_scores

#Step 3:Decision Model using Linear Regression:

# calculate the predicted sentiment score and decide to buy/sell/hold for each day
def decision_model(scores, sell_threshold, buy_threshold):
    # Sort scores by date
    scores.sort(key=lambda x: datetime.strptime(x["Date"], '%Y-%m-%d'))

    # isolate dates and scores
    dates = [score["Date"] for score in scores]
    sentiment_scores = [score["Score"] for score in scores]

    # turn days into a range of numbers and scores into an array
    enumerated_dates = np.array(range(len(sentiment_scores))).reshape(-1, 1)
    score_array = np.array(sentiment_scores)

    #initialize linear regression model
    model = LinearRegression()

    # fit the model to be trained off the days while targeting the scores
    model.fit(enumerated_dates, score_array)

    # the slope is the coefficient of the dates which is how much every day increases the predicted sentiment score
    slope = model.coef_[0]

    # initialize a list of buy/sell/hold decisions
    buy_hold_sell = []

    for i, (date, score) in enumerate(zip(dates, sentiment_scores)):
        # need at least two points to calculate slope
        if i > 1:
            # current time span
            current_days = np.array(range(i+1)).reshape(-1, 1)

            # current scores for the current time span
            current_scores = np.array(sentiment_scores[:i+1])

            # refit the model
            model.fit(current_days, current_scores)

            # recalculate slope for each day using data up to that day
            slope = model.coef_[0]

        # if the trend is positive and the scire meets the threshold
        if slope > 0 and score >= buy_threshold:
            decision = "BUY"
        elif slope < 0 and score <= sell_threshold:
            decision = "SELL"
        else:
            decision = "HOLD"

        # log the date, score, trade decision, thresholds, and the sentiment slope
        buy_hold_sell.append({
            "Date": date,
            "Sentiment_Score": score,
            "Action_Taken": decision,
            "Buy_Threshold": buy_threshold,
            "Sell_Threshold": sell_threshold,
            "Slope": slope
        })
    print_latest_buy_hold_sell(buy_hold_sell)
    return buy_hold_sell

def print_latest_buy_hold_sell(buy_hold_sell):
    item = buy_hold_sell[-1]
    print("Final Trade Decision Log:")
    print("\n----------------------------------------------------------")
    print("Date: ", item["Date"])
    print("Sentiment Score: ", item["Sentiment_Score"])
    print("Action Taken: ", item["Action_Taken"])
    print("Buy Threshold: ", item["Buy_Threshold"])
    print("Sell Threshold: ", item["Sell_Threshold"])
    print("Slope: ", item["Slope"])
    print("----------------------------------------------------------\n")

#Step 4: Backtesting Framework Functions

#Return on Investment: Measures the overall gain or loss relative to the initial investment.
# calculates and returns the return on investment based on a given portfolio and starting capital
def calculate_return_on_investment(portfolio, starting_capital):
    ROI = portfolio[-1]["Net_Profits"] / starting_capital
    return ROI

#Win/Loss Ratio: Tracks the number of profitable trades versus losing trades.
# calculates and returns the win/loss ratio of a portfolio when given a portfolio
def calculate_win_loss_ratio(portfolio):
    wins = 0
    losses = 0

    # assume "trade_history" logs each trade outcome in context
    for day in portfolio:
        if day["Profit_Today"] > 0:
            wins += 1
        elif day["Profit_Today"] < 0:
            losses += 1
    win_loss_ratio = wins / losses if losses > 0 else 1
    return win_loss_ratio

#Sharpe Ratio: Measures risk-adjusted return, useful for comparing different strategies.
# calculates and returns the sharpe ratio of a portfolio when given a portfolio
def calculate_sharpe_ratio(portfolio):
    daily_returns = [portfolio[i]["Profit_Today"] for i in range(1, len(portfolio))]
    average_return = np.mean(daily_returns)
    std_dev = np.std(daily_returns)
    sharpe_ratio = average_return / std_dev if std_dev != 0 else 0
    return sharpe_ratio

# Maximum Drawdown: Reflects the largest peak-to-trough decline, showing the worst-case scenario for the strategy.
# calculates and returns the maximum drawdown of a portfolio when given a portfolio
def calculate_max_drawdown(portfolio):
    max_drawdown = 0
    peak = portfolio[0]["Current_Portfolio_Value"]

    for day in portfolio:
        if day["Current_Portfolio_Value"] > peak:
            peak = day["Current_Portfolio_Value"]
        drawdown = (peak - day["Current_Portfolio_Value"]) / peak
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown

# Backtracking System: The system by which the prediction system is simulated against the historical open/close prices of the stock.
def backtracking_system(starting_capital, buy_threshold, sell_threshold, daily_open_close, daily_sentiment_scores):
    # initialize basic portfolio varables
    stock_holdings = 0
    cash = starting_capital
    buy_count = 0
    sell_count = 0

    # initialize list for tracking portfolio over time
    portfolio = []

    # sort these by date to ensure they are in chronological order
    daily_open_close.sort(key=lambda x: x[0])
    daily_sentiment_scores.sort(key=lambda x: x["Date"])

    # create a list of sentiment scores for each date in the dailyOpenClose data
    sentiment_scores = []
    for open_close_day in daily_open_close:
        # find the sentiment score for matching dates in dailyOpenClose
        sentiment_score = next((score["Score"] for score in daily_sentiment_scores if score["Date"] == open_close_day[0]), 0)
        sentiment_scores.append(sentiment_score)

    for i, day in enumerate(daily_open_close):
        # grab the date from dailyOpenClose
        date = day[0]

        # grab the stock opening and closing prices
        open_price = float(day[1])
        close_price = float(day[2])

        # stock price is set to the average of open and close price
        stock_price = (open_price + close_price) / 2

        if i > 1:
            # current time span
            current_days = np.array(range(i + 1)).reshape(-1, 1)

            # current scores for the current time span
            current_scores = np.array(sentiment_scores[: i + 1])

            # initialize the linear regression model
            model = LinearRegression()

            # refit the model
            model.fit(current_days, current_scores)

            # recalculate slope for each day using data up to that day
            slope = model.coef_[0]
        else:
            # no slope for the first two days as it needs two
            slope = 0

        # the action that is being decided on, HOLD by default and changes based on slope and thresholds
        action = "HOLD"

        # get current day's sentiment score
        current_score = sentiment_scores[i]

        # buy/sell 1 unit of stock if there is sufficient cash or stocks in possesion and meets slope/threshold values
        # track the number of stock owned and the number of buy/sell actions
        if slope > 0 and current_score >= buy_threshold and cash >= stock_price:
            stock_holdings += 1
            cash -= stock_price
            buy_count += 1
            action = "BUY"
        elif slope < 0 and current_score <= sell_threshold and stock_holdings > 0:
            stock_holdings -= 1
            cash += stock_price
            sell_count += 1
            action = "SELL"

        # calculate the daily portfolio value and append to list with logs of all relevant info
        current_portfolio_value = cash + (stock_holdings * stock_price)

        # get the previous portfolio valus
        previous_portfolio_value = portfolio[-1]["Current_Portfolio_Value"] if portfolio else starting_capital

        # log all relevant portfolio information
        portfolio.append({
            "Date": date,
            "Current_Portfolio_Value": current_portfolio_value,
            "Current_Stock_Price": stock_price,
            "Stock_Held": stock_holdings,
            "Cash_Held": cash,
            "Slope": slope,
            "Sentiment_Score": current_score,
            "Buy_Threshold": buy_threshold,
            "Sell_Threshold": sell_threshold,
            "Action_Taken": action,
            "Buy_Count": buy_count,
            "Sell_Count": sell_count,
            "Profit_Today": current_portfolio_value - previous_portfolio_value,
            "Net_Profits": current_portfolio_value - starting_capital,
            })

    # display final stats of the backtesting run
    print_latest_portfolio(portfolio)
    ROI = calculate_return_on_investment(portfolio, starting_capital)
    print("\nThe Return on Investment was", ROI, "times the starting capital.")
    WLR = calculate_win_loss_ratio(portfolio)
    print("\nThe Win/Loss Ratio was", WLR)
    SR = calculate_sharpe_ratio(portfolio)
    print("\nThe Sharpe Ration was", SR)
    MD = calculate_max_drawdown(portfolio)
    print("\nThe Maximum Drawdown was", MD)

    return portfolio

def print_latest_portfolio(portfolio):
    item = portfolio[-1]
    print("Final Portfolio Log:")
    print("\n----------------------------------------------------------")
    print("Date: ", item["Date"])
    print("Current Portfolio Value: $", f"{item['Current_Portfolio_Value']:.2f}")
    print("Current Stock Price: $", f"{item['Current_Stock_Price']:.2f}")
    print("Stock Held: ", item["Stock_Held"])
    print("Cash Held: $", f"{item['Cash_Held']:.2f}")
    print("Slope: ", item["Slope"])
    print("Sentiment Score: ", item["Sentiment_Score"])
    print("Buy Threshold: ", item["Buy_Threshold"])
    print("Sell Threshold: ", item["Sell_Threshold"])
    print("Action Taken: ", item["Action_Taken"])
    print("Buy Count: ", item["Buy_Count"])
    print("Sell Count: ", item["Sell_Count"])
    print("Profit Today: $", f"{item['Profit_Today']:.2f}")
    print("Net Profits: $", f"{item['Net_Profits']:.2f}")
    print("----------------------------------------------------------\n")

# Step 5: Visualize
# plots and displays the sentiment scores, trading decisions, stock prices, and various other portfolio metrics over time
def visualize_data(portfolio):
    # use pandas to turn portfolio to a Dataframe for better plotting
    df = pd.DataFrame(portfolio)

    # convert Date to datetime for better plotting
    df['Date'] = pd.to_datetime(df['Date'])

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # plot the portfolio value over time on the left y-axis
    ax1.plot(df['Date'], df['Current_Portfolio_Value'], label='Portfolio Value', color='blue', linewidth=2)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # ensure the axis format keeps the numbers in correct format
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax1.grid()

    # overloyed with the first plot and using the right y-axis, is the stock price over time
    ax2 = ax1.twinx()
    ax2.plot(df['Date'], df['Current_Stock_Price'], label='Stock Price', color='orange', linewidth=2)
    ax2.set_ylabel('Stock Price', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # ensure the axis format keeps the numbers in correct format
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:,.2f}'))

    plt.title('Portfolio Value and Stock Price Over Time')
    fig.tight_layout()
    plt.savefig("stock_price_portfolio_value.jpg", format="jpg", dpi=300)
    plt.show()
    plt.close()

    # plot a scatter plot of the sentiment scores over time
    plt.figure(figsize=(12, 8))
    plt.scatter(df['Date'], df['Sentiment_Score'], label='Sentiment Score', color='green')
    plt.title('Daily Sentiment Scores Over Time')
    plt.xlabel('Date')
    plt.ylabel('Daily Sentiment Score')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("sentiment_score.jpg", format="jpg", dpi=300)
    plt.show()
    plt.close()

    # plots the change in stock price over time except it changes color by trade decisions
    plt.figure(figsize=(12, 8))
    for action in ['BUY', 'SELL', 'HOLD']:
        subset = df[df['Action_Taken'] == action]
        if action == 'BUY':
            plt.plot(subset['Date'], subset['Current_Stock_Price'],
                     label=f'Stock Price and {action}ing', color='green', linestyle='-', marker='o', markersize=6, linewidth=2)
        elif action == 'SELL':
            plt.plot(subset['Date'], subset['Current_Stock_Price'],
                     label=f'Stock Price and {action}ing', color='red', linestyle='-', marker='x', markersize=6, linewidth=2)
        else:
            plt.plot(subset['Date'], subset['Current_Stock_Price'],
                     label=f'Stock Price and {action}ing', color='blue', linestyle='-', alpha=0.6)
    plt.title('Stock Price Over Time with Actions')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("trade_actions_stock_price.jpg", format="jpg", dpi=300)
    plt.show()
    plt.close()

# Assistant Main Functions
# function to prompt the user for input for sentiment thresholds of the decision making system and backstracking system
def get_threshold(prompt):
    while True:
        try:
            # convert input to float
            value = float(input(prompt))
            # check if the value is in the acceptable range
            if -100 <= value <= 100:
                return value
            else:
                print("Error: Please enter a value between -100 and 100.")
        except ValueError:
            print("Error: Invalid input. Please enter a numeric value.")

# function to prompt the user for input for the starting capital of the backtracking system

def get_capital(prompt):
    while True:
        try:
            # convert input to int
            value = int(input(prompt))
            # check if the value is in the acceptable range
            if 0 <= value <= 1000000000:
                return value
            else:
                print("Error: Please enter a value between 0 and 1,000,000,000.")
        except ValueError:
            print("Error: Invalid input. Please enter a numeric value.")

# Main Prep function for Steps 1 and 2. Run this before main. Only run once due to heavy wait times and API call usage.
def main_prep():
    # globally declare so they can be called in main
    global daily_open_close, tweet_list, analysis, daily_sentiment_scores

    print("Step 1 Starting...")

    print("Getting Historical Stock Prices...")
    daily_open_close = get_daily_open_close_prices()

    print("Assembling Tweets...")
    tweet_list = format_tweets()

    print("Step 1 Done...\n")

    print("Step 2 Starting...")

    print("Analyzing Tweet Sentiments...")
    print("This may take time, please wait...")
    analysis = analyze_tweet_sentiment(tweet_list)

    print("Calculating Daily Sentiment Scores...")
    daily_sentiment_scores = calculate_daily_sentiment_scores(analysis)

    print("Step 2 Done...\n")

# MAIN FUNCTION
# Main function for Steps 3, 5, and 5
def main():
    # either call or avoid main prep if running multiple times so as to not waste time on analysis and API calls
    # option to run analysis, only accepts Y/N
    run = True;
    while run:
        user_input = input("\nDo you want to run SENTIMENT ANALYSIS SYSTEM? (Y/N): ").strip().upper()
        if user_input == 'N':
            print("\nSkipping Sentiment Analysis.")
            run = False
        elif user_input == 'Y':
            print("\nRunning Sentiment Analysis.")
            main_prep()
        else:
            print("Error: Invalid input. Please enter Y or N.")

    print("Step 3 Starting...")

    print("\n - BUY/HOLD/SELL DECISION MODEL SYSTEM - \n")
    while True:
        # get thresholds from user input
        sell_threshold = get_threshold("Please Input Sentiment Threshold of Selling Point (-100 to 100): ")
        buy_threshold = get_threshold("Please Input Sentiment Threshold of Buying Point (-100 to 100): ")

        # run system
        buy_hold_sell = decision_model(daily_sentiment_scores, sell_threshold, buy_threshold)

        # option to try the system again, only accepts Y/N
        user_input = input("\nDo you want to run BUY/HOLD/SELL DECISION MODEL SYSTEM again? (Y/N): ").strip().upper()
        if user_input == 'N':
            print("\nExiting BUY/HOLD/SELL DECISION SYSTEM loop.")
            break
        elif user_input != 'Y':
            print("\nInvalid input. Exiting BUY/HOLD/SELL DECISION MODEL SYSTEM loop.")
            break

    # export decision model output to as a JSON file called "buy_hold_sell.json"
    with open("buy_hold_sell.json", "w") as file:
        json.dump(buy_hold_sell, file, indent=4)

    # confirm json export
    print("\nData exported to 'buy_hold_sell.json'")

    print("\nStep 3 Done...")

    print("\nStep 4 Starting...")

    print("\n - BACKTRACKING SYSTEM - \n")
    while True:
        # get thresholds and starting capital from user input
        sell_threshold = get_threshold("Please Input Sentiment Threshold of Selling Point (-100 to 100): ")
        buy_threshold = get_threshold("Please Input Sentiment Threshold of Buying Point (-100 to 100): ")
        starting_capital = get_capital("Please Input the Amount of Starting Capital (0 to 1,000,000,000): ")

        # run system
        portfolio = backtracking_system(starting_capital, buy_threshold, sell_threshold, daily_open_close, daily_sentiment_scores)

        # option to try the system again, only accepts Y/N
        user_input = input("\nDo you want to run BACKTRACKING SYSTEM again? (Y/N): ").strip().upper()
        if user_input == 'N':
            print("\nExiting BACKTRACKING SYSTEM loop.")
            break
        elif user_input != 'Y':
            print("\nInvalid input. Exiting BACKTRACKING SYSTEM loop.")
            break

    # export portfolio list as a JSON file called "portfolio_values.json"
    with open("portfolio.json", "w") as file:
        json.dump(portfolio, file, indent=4)

    # confirm json export
    print("\nData exported to 'portfolio.json'")

    print("\nStep 4 Done...")

    print("\nStep 5 Starting...")

    print("\nVisualizing Latest Portfolio Performance...")

    visualize_data(portfolio)

    print("\nStep 5 Done...")

# RUNS MAIN FUNCTION
if __name__=="__main__":
    main()