from src.TextSummarizer.logging import logger
from src.TextSummarizer.pipeline.stage_1_data_ingestion import DataIngestionPipeline

logger.info("Logging setup complete.")

STAGE_NAME = "Data Ingestion"

try:
    logger.info(f"stage {STAGE_NAME} started")
    data_ingestion_pipeline = DataIngestionPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e