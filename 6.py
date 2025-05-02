import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

date_range = pd.date_range(end=pd.Timestamp.today(), periods=50, freq='D')
rsi_values = np.random.uniform(20, 80, size=50)

df = pd.DataFrame({
    'time': date_range,
    'RSI': rsi_values
})

df.to_csv("indicators_output.csv", index=False)

df = pd.read_csv("indicators_output.csv")
df['time'] = pd.to_datetime(df['time'])

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.bar(df['time'], df['RSI'], color='skyblue')
plt.title("bar - RSI 14")
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
plt.scatter(df['time'], df['RSI'], color='orange', s=10)
plt.title("scatter - RSI 14")
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
plt.plot(df['time'], df['RSI'], color='green')
plt.title("plot - RSI 14")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("rsi_visualization.png")
plt.show()
