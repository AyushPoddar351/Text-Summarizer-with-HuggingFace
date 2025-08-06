from src.TextSummarizer.config.configuration import ConfigurationManager
from src.TextSummarizer.components.model_trainer import ModelTrainer
from src.TextSummarizer.logging import logger

class ModelTrainerPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):

        config=ConfigurationManager()
        model_training_config=config.get_model_trainer_config()
        model_training=ModelTrainer(config=model_training_config)
        model_training.train()
