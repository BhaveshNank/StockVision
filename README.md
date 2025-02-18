## StockVision 📊  
A Machine Learning & Sentiment Analysis-based Dashboard for analyzing stock trends, forecasting future prices, and monitoring market sentiment.

## **🔧 Installation & Setup**  

### **1️⃣ Clone the Repository**  

git clone https://github.com/your-username/StockVision.git
cd StockVision

2️⃣ Create & Activate Virtual Environment
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Set Up API Keys
StockVision requires API keys for fetching real-time stock data and news.

🔹 Finnhub API (Stock Data & News)
Sign up at Finnhub to get an API Key.
Create a .env file in the project root and add:

  FINNHUB_API_KEY=your_api_key_here

Or set it manually in the terminal:

export FINNHUB_API_KEY=your_api_key_here  # macOS/Linux
set FINNHUB_API_KEY=your_api_key_here    # Windows

5️⃣ Run the Dashboard

python app.py
Open http://127.0.0.1:8050/ in your browser.

### **Features**

### **Homepage**
Shows the input section, stock details, graph visualization, latest news and news sentiment analysis.
![Dashboard Homepage](assets/Homepage.png)

### **Stock Details**
Typing a Stock code will Display the stock information and clicking on the stock price will display the price trends.
![Stock Details](assets/stock_details.png)

### **Technical Indicators**
Select RSI, MACD, or Bollinger Bands for stock analysis.
![Indicators](assets/indicators.png)

### **Forecasting Stock Prices**
Predict future stock trends using machine learning.
![Stock Forecast](assets/forecast.png)

### **Latest News on Selected Stock**
Get real-time news headlines related to the stock.
![Latest News](assets/latest_news.png)

### **Sentiment Analysis Based on News**
Analyze market sentiment using AI-powered news insights.  
This feature fetches the latest stock-related news and uses **VADER Sentiment Analysis** (Natural Language Processing) to determine whether the sentiment is **positive, neutral, or negative**.  

#### **📊 Sentiment Gauge**
- A **gauge meter** visually represents the sentiment score.
- **Ranges from -1 to 1**:
  - 🔴 **Negative Sentiment (-1 to -0.3)** → Bearish Market 🐻
  - 🟡 **Neutral Sentiment (-0.3 to 0.3)** → Uncertain Market ⚖️
  - 🟢 **Positive Sentiment (0.3 to 1)** → Bullish Market 🐂
- The gauge dynamically updates based on **real-time news sentiment**.
![Sentiment Analysis](assets/sentiment_analysis.png)
