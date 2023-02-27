import eikon as ek
import os
import datetime

import pandas as pd

# html.Div(
#             [
#
#             ]
#         )

# create a sample dataframe
df = pd.DataFrame({'A': [1, 2, 3, 4, 5,6], 'B': ['a', 'b', 'c', 'd', 'e', 'f']})

# select all rows except for the last row
result = df
print(result)