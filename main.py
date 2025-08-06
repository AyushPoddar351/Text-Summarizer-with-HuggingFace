from src.TextSummarizer.logging import logger
from src.TextSummarizer.pipeline.stage_1_data_ingestion import DataIngestionPipeline
from src.TextSummarizer.pipeline.stage_2_data_transformation import DataTransformationPipeline
from src.TextSummarizer.pipeline.stage_3_model_trainer import ModelTrainerPipeline
from src.TextSummarizer.pipeline.stage_4_model_evaluation import ModelEvaluationPipeline

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

STAGE_NAME = "Data Transformation"

try:
    logger.info(f"stage {STAGE_NAME} started")
    data_transformation_pipeline = DataTransformationPipeline()
    data_transformation_pipeline.initiate_data_transformation()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Trainer"

try:
    logger.info(f"stage {STAGE_NAME} started")
    model_training_pipeline = ModelTrainerPipeline()
    model_training_pipeline.initiate_model_training()
    logger.info(f"stage {STAGE_NAME} completed")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evaluation stage"
try: 
   logger.info(f"*******************")
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evaluation = ModelEvaluationPipeline()
   model_evaluation.initiate_model_evaluation()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e
