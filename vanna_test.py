import os, sys
from vanna_setup import vn

question = "Which course has the lowest price?"

# Suppress Vanna's internal prints
sys.stdout = open(os.devnull, 'w')
sql = vn.generate_sql(question=question)
sys.stdout = sys.__stdout__

df = vn.run_sql(sql=sql)
print("SQL:", sql)
print(df)