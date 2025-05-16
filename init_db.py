from dotenv import load_dotenv

from db import setup_database

load_dotenv()

if __name__ == "__main__":
    setup_database()
