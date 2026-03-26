import pandas as pd

file_path = 'spy_minutes_2024.csv'
output_list = []

# 1. Process in chunks to keep memory usage low
for chunk in pd.read_csv(file_path, chunksize=100000):
    
    # 2. Convert nanoseconds to datetime (UTC by default)
    chunk['dt_utc'] = pd.to_datetime(chunk['window_start'], unit='ns', utc=True)
    
    # 3. Convert to NYSE Local Time (handles Daylight Savings automatically)
    chunk['dt_ny'] = chunk['dt_utc'].dt.tz_convert('US/Eastern')
    
    # 4. Filter for the 9:30 - 9:35 AM window
    # We use .between_time for cleaner syntax
    chunk.set_index('dt_ny', inplace=True)
    filtered = chunk.between_time('09:30', '09:34') # 09:34 is the start of the 5th candle
    
    output_list.append(filtered.reset_index())

# Combine windows
df = pd.concat(output_list)

# 5. Group by date to verify you have exactly 5 candles per day
daily_counts = df.groupby(df['dt_ny'].dt.date).size()
print(f"Processed {len(daily_counts)} trading days.")