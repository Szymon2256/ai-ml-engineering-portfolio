import pandas as pd
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        """
        Load data from a JSON file and return it as a pandas DataFrame.
        """
        try:
            df = pd.read_csv(self.file_path)
            logger.info(f" Data loaded successfully from {self.file_path}")
            return df
        except Exception as e:
            logger.error(f" Error loading data from {self.file_path}: {e}")
            raise
    
    def get_clean_ticket_dataset(self) -> pd.DataFrame:
        """
        Load the dataset and perform basic cleaning operations.
        """
        df = self.load_data()
        
        df.drop(columns=['tag_9'], inplace=True)
        df.loc[df["business_type"].isin(["Pit Services", "Adobe Photoshop 2024", "_IT_Services_", "IT Consulting Service"]), "business_type"] = "IT Services"

        # Connecting tags into a single column and creating a new column for the combined context of the problem
        tags = [col for col in df.columns if col.startswith('tag_')]
        df["tags"] = df[tags].apply(lambda x: x.dropna().tolist(), axis=1)

        # Combining 'subject' and 'body' into a single column 'context_problem'
        df["context_problem"] = df["subject"].fillna("") + " " + df["body"].fillna("")
        df_with_tags = df.drop(columns=tags + ['subject', 'body'])
        
        # Reordering columns to have 'context_problem' first
        new_order = ['context_problem'] + df_with_tags.columns[:-1].to_list()
        df_with_tags = df_with_tags[new_order]

        logger.info(" Data cleaned successfully")
        
        return df_with_tags
        
