import os

from dotenv import load_dotenv

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

load_dotenv()

DATABASE_URL = os.getenv("POSTGRES_URL")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)

class SessionManager:
    session: Session | None = None

    def __new__(cls):
        # check if instance already exists and return
        if not hasattr(cls, 'instance'):
            print('new session started')
            cls.instance = super(SessionManager, cls).__new__(cls)
        return cls.instance
    
    def __init__(self):
        if not self.session:
            self.session = SessionLocal()

    def __del__(self):
        self.close_session()

    def close_session(self):
        print('I am closing')
        self.session.close()

Base = declarative_base()