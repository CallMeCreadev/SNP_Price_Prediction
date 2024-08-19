from temporary_data_storage import TemporaryDataStorage

storage = TemporaryDataStorage("stock_closings.pkl")
stock_data = storage.load_data()
print(stock_data['AAPL'])
