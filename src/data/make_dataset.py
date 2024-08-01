import os
import sys
import zipfile
from src.logger import logging
from src.exception import CustomException
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

os.environ['KAGGLE_USERNAME'] = os.getenv('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = os.getenv('KAGGLE_KEY')

@dataclass
class Dataingestionconfig:
    raw_data_path:Path=Path('data/raw')
    processed_data_path:Path=Path('data/processes')
    


class DataIngestion:
    def __init__(self) :
        self.ingestion_confi=Dataingestionconfig()
        
    def initiate_data_injestion(self):
        logging.info("Data injestion has Started")
        try:
            logging.info('Checking existance')
            path=self.ingestion_confi.raw_data_path
            # os.system(f"kaggle datasets download -d kapillondhe/american-sign-language -p {path}")
            with zipfile.ZipFile(Path('data/raw/american-sign-language.zip'),'r') as zip_:
                zip_.extractall(self.ingestion_confi.processed_data_path)
            return self.ingestion_confi.processed_data_path
        except Exception as e:
            raise CustomException(e,sys)
            

if __name__=="__main__":
    a=DataIngestion().initiate_data_injestion()



