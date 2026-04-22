import os
from dotenv import load_dotenv
from alpaca_trade_api.rest import REST

load_dotenv("config/secrets.env")

api = REST(
    os.getenv("ALPACA_API_KEY"),
    os.getenv("ALPACA_SECRET_KEY"),
    os.getenv("ALPACA_BASE_URL"),
    api_version='v2'
)

account = api.get_account()
print("Account status:", account.status)