import os
from dotenv import load_dotenv

ENV = os.getenv("ENV")
load_dotenv(".env")

if ENV:
    ENV_FILE = f".env.{ENV}"
    if os.path.exists(ENV_FILE):
        load_dotenv(ENV_FILE, override=True)
