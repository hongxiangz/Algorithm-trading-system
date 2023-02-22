import eikon as ek
import os

eikon_api = os.getenv('EIKON_API')
ek.set_app_key(eikon_api)

df = ek.get_timeseries(["MSFT.O"],
                       start_date="2016-01-01",
                       end_date="2023-01-19")
print(df)