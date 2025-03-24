import yfinance as yf

# Define stock tickers for each company.
companies = {
    "apple": "AAPL",
    # "microsoft": "MSFT",
    # "intel": "INTC",
    # "nvidia": "NVDA"
}

# Define the date range for the historical data.
start_date = "2010-01-01"
end_date = "2023-12-31"

for company, ticker in companies.items():
    print(f"Downloading data for {company} ({ticker})...")
    # Download the historical data from Yahoo Finance.
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        print(f"Warning: No data downloaded for {company}.")
    else:
        # Save the data to a CSV file.
        output_filename = f"{company}.csv"
        df.to_csv(output_filename)
        print(f"Data for {company} saved to {output_filename}.")