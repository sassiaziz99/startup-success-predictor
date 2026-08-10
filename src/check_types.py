import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))
rounds = pd.read_sql("SELECT DISTINCT funding_round_type FROM funding_rounds", engine)
print(rounds)