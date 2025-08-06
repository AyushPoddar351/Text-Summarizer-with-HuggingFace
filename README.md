# Text Summarization System - PEGASUS Transformer API

An advanced natural language processing system built with Hugging Face Transformers and FastAPI that performs high-quality abstractive text summarization using Google's PEGASUS model, featuring complete MLOps pipeline with automated training, evaluation, and deployment capabilities.

## 🎯 Project Overview

This project implements a production-ready text summarization service leveraging state-of-the-art transformer architecture (PEGASUS) fine-tuned on conversation data. The system provides an end-to-end MLOps pipeline from data ingestion through model deployment, featuring automated training workflows, comprehensive evaluation metrics, and RESTful API endpoints for real-time text summarization.

## 🚀 Key Features

### Natural Language Processing
- **PEGASUS Model**: Google's pre-trained transformer for abstractive summarization
- **Fine-tuning Pipeline**: Custom training on SAMSum conversation dataset
- **Sequence-to-Sequence Architecture**: Advanced encoder-decoder framework
- **Beam Search Decoding**: Optimized text generation with multiple hypotheses
- **Length Penalty Optimization**: Balanced summary length control

### MLOps Pipeline
- **Automated Data Ingestion**: Dataset download and preprocessing
- **Data Transformation**: Tokenization and feature engineering
- **Model Training**: Distributed training with Hugging Face Trainer
- **Model Evaluation**: ROUGE metrics and performance assessment
- **Model Serving**: FastAPI deployment with prediction endpoints

### Production Features
- **RESTful API**: FastAPI framework for scalable web service
- **Batch Processing**: Efficient handling of multiple text inputs
- **Model Persistence**: Automatic model and tokenizer saving
- **Configuration Management**: YAML-based parameter management
- **Logging System**: Comprehensive operation tracking

## 📊 Model Architecture

### PEGASUS Transformer
```python
# Pre-trained Model Configuration
model_name = "google/pegasus-cnn_dailymail"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Fine-tuning Parameters
max_input_length = 1024    # Maximum input sequence length
max_target_length = 128    # Maximum summary length
num_beams = 8             # Beam search width
length_penalty = 0.8      # Length normalization factor
```

### Training Configuration
```python
TrainingArguments:
  num_train_epochs: 1
  warmup_steps: 500
  per_device_train_batch_size: 1
  weight_decay: 0.01
  logging_steps: 10
  evaluation_strategy: steps
  eval_steps: 500
  gradient_accumulation_steps: 16
```

## 🏗️ System Architecture

```
TextSummarizer/
├── src/TextSummarizer/
│   ├── components/
│   │   ├── data_ingestion.py         # Dataset download and extraction
│   │   ├── data_transformation.py    # Tokenization and preprocessing
│   │   ├── model_trainer.py          # Model fine-tuning pipeline
│   │   └── model_evaluation.py       # Performance evaluation
│   ├── pipeline/
│   │   ├── stage_1_data_ingestion.py    # Data ingestion pipeline
│   │   ├── stage_2_data_transformation.py # Preprocessing pipeline
│   │   ├── stage_3_model_trainer.py      # Training pipeline
│   │   ├── stage_4_model_evaluation.py   # Evaluation pipeline
│   │   └── prediction_pipeline.py        # Inference pipeline
│   ├── config/
│   │   └── configuration.py          # Configuration management
│   ├── entity/
│   │   └── __init__.py              # Data classes and entities
│   └── utils/
│       └── common.py                # Utility functions
├── config/
│   └── config.yaml                  # System configuration
├── params.yaml                      # Training parameters
├── main.py                          # Training pipeline orchestrator
└── app.py                          # FastAPI application server
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- FastAPI & Uvicorn
- CUDA (optional, for GPU training)

### 1. Clone Repository
```bash
git clone <repository-url>
cd text-summarizer
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install torch transformers datasets
pip install fastapi uvicorn
pip install evaluate rouge-score
pip install box-python pathlib ensure pyyaml
```

### 4. Configure System
```yaml
# config/config.yaml
data_ingestion:
  source_URL: https://github.com/krishnaik06/datasets/raw/refs/heads/main/summarizer-data.zip
  
model_trainer:
  model_ckpt: google/pegasus-cnn_dailymail
  
model_evaluation:
  metric_file_name: artifacts/model_evaluation/metrics.csv
```

### 5. Run Training Pipeline
```bash
python main.py
```

### 6. Start API Server
```bash
python app.py
# or
uvicorn app:app --host 0.0.0.0 --port 8080
```

## 📈 MLOps Pipeline Stages

### Stage 1: Data Ingestion
```python
class DataIngestion:
    def download_file(self):
        """Download SAMSum dataset from remote source"""
        filename, headers = request.urlretrieve(
            url=self.config.source_URL,
            filename=self.config.local_data_file
        )
        
    def extract_zip_file(self):
        """Extract dataset to working directory"""
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
```

### Stage 2: Data Transformation
```python
class DataTransformation:
    def convert_examples_to_features(self, example_batch):
        """Convert dialogue-summary pairs to model inputs"""
        input_encodings = self.tokenizer(
            example_batch['dialogue'], 
            max_length=1024, 
            truncation=True
        )
        
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                example_batch['summary'], 
                max_length=128, 
                truncation=True
            )
        
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
```

### Stage 3: Model Training
```python
class ModelTrainer:
    def train(self):
        """Fine-tune PEGASUS model on SAMSum dataset"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_ckpt
        ).to(device)
        
        trainer = Trainer(
            model=model_pegasus,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"]
        )
        
        trainer.train()
```

### Stage 4: Model Evaluation
```python
class ModelEvaluation:
    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer):
        """Evaluate model performance using ROUGE metrics"""
        for article_batch, target_batch in tqdm(zip(article_batches, target_batches)):
            summaries = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                length_penalty=0.8,
                num_beams=8,
                max_length=128
            )
            
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True) 
                               for s in summaries]
            metric.add_batch(predictions=decoded_summaries, references=target_batch)
        
        return metric.compute()
```

## 🌐 FastAPI Application

### API Endpoints

#### Health Check
```python
@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")
```

#### Training Endpoint
```python
@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")
```

#### Prediction Endpoint
```python
@app.post("/predict")
async def predict_route(text):
    try:
        obj = PredictionPipeline()
        summary = obj.predict(text)
        return summary
    except Exception as e:
        raise e
```

### Prediction Pipeline
```python
class PredictionPipeline:
    def predict(self, text):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        gen_kwargs = {
            "length_penalty": 0.8, 
            "num_beams": 8, 
            "max_length": 128
        }
        
        pipe = pipeline(
            "summarization", 
            model=self.config.model_path,
            tokenizer=tokenizer
        )
        
        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        return output
```

## 📊 Performance Evaluation

### ROUGE Metrics
```python
# Evaluation metrics calculation
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_metric = evaluate.load('rouge')

score = self.calculate_metric_on_test_ds(
    dataset_samsum_pt['test'][0:10], 
    rouge_metric, 
    model_pegasus, 
    tokenizer, 
    batch_size=2, 
    column_text='dialogue', 
    column_summary='summary'
)

# ROUGE score analysis
rouge_dict = {rn: score[rn] for rn in rouge_names}
```

### Model Performance Metrics
- **ROUGE-1**: Unigram overlap between generated and reference summaries
- **ROUGE-2**: Bigram overlap measuring fluency and coherence
- **ROUGE-L**: Longest common subsequence for structural similarity
- **ROUGE-Lsum**: Summary-level longest common subsequence

## 🔧 Configuration Management

### System Configuration
```yaml
# config/config.yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: https://github.com/krishnaik06/datasets/raw/refs/heads/main/summarizer-data.zip
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

data_transformation:
  root_dir: artifacts/data_transformation
  data_path: artifacts/data_ingestion/samsum_dataset
  tokenizer_name: google/pegasus-cnn_dailymail

model_trainer:
  root_dir: artifacts/model_trainer
  data_path: artifacts/data_transformation/samsum_dataset
  model_ckpt: google/pegasus-cnn_dailymail

model_evaluation:
  root_dir: artifacts/model_evaluation
  data_path: artifacts/data_transformation/samsum_dataset
  model_path: artifacts/model_trainer/pegasus-samsum-model
  tokenizer_path: artifacts/model_trainer/tokenizer
  metric_file_name: artifacts/model_evaluation/metrics.csv
```

### Training Parameters
```yaml
# params.yaml
TrainingArguments:
  num_train_epochs: 1
  warmup_steps: 500
  per_device_train_batch_size: 1
  weight_decay: 0.01
  logging_steps: 10
  evaluation_strategy: steps
  eval_steps: 500
  save_steps: 1e6
  gradient_accumulation_steps: 16
```

## 🚀 Deployment Options

### Local Development
```bash
# Start FastAPI server
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Cloud Deployment
```bash
# Build and deploy to cloud platforms
docker build -t text-summarizer .
docker run -p 8080:8080 text-summarizer

# Or deploy directly to cloud services
# AWS, GCP, Azure container services
```

## 🧪 Usage Examples

### API Usage
```python
import requests

# Text summarization request
text = """
Person A: Hey, how was your day at work today?
Person B: It was really busy! We had three back-to-back meetings about the new project launch. 
The marketing team presented their campaign strategy, and we discussed the timeline for implementation.
Person A: That sounds intense. What's the project about?
Person B: We're launching a new mobile app for food delivery. 
It's been in development for eight months, and we're finally ready to go to market next quarter.
"""

response = requests.post(
    "http://localhost:8080/predict",
    json={"text": text}
)

summary = response.json()
print(f"Summary: {summary}")
```

### Direct Pipeline Usage
```python
from src.TextSummarizer.pipeline.prediction_pipeline import PredictionPipeline

# Initialize prediction pipeline
predictor = PredictionPipeline()

# Generate summary
text = "Your long text here..."
summary = predictor.predict(text)
print(f"Generated Summary: {summary}")
```

## 📈 Advanced Features

### Batch Processing
```python
def batch_summarize(texts, batch_size=4):
    """Process multiple texts efficiently"""
    summaries = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_summaries = [predictor.predict(text) for text in batch]
        summaries.extend(batch_summaries)
    return summaries
```

### Custom Fine-tuning
```python
# Fine-tune on domain-specific data
def custom_fine_tune(custom_dataset_path):
    config = ConfigurationManager()
    config.config.data_transformation.data_path = custom_dataset_path
    
    # Run training pipeline with custom data
    model_trainer = ModelTrainerPipeline()
    model_trainer.initiate_model_training()
```

## 🔍 Performance Optimization

### GPU Acceleration
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Optimized inference
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs["input_ids"].to(device),
        attention_mask=inputs["attention_mask"].to(device),
        max_length=128,
        num_beams=8,
        length_penalty=0.8,
        early_stopping=True
    )
```

### Memory Management
```python
# Gradient accumulation for large batches
gradient_accumulation_steps = 16
per_device_train_batch_size = 1

# Automatic mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss / gradient_accumulation_steps
```

## 🔮 Future Enhancements

### Model Improvements
- **Multi-language Support**: Extend to multiple languages
- **Domain Adaptation**: Specialized models for different domains
- **Extractive Summarization**: Hybrid extractive-abstractive approach
- **Real-time Streaming**: Live text summarization capabilities

### System Enhancements
- **Caching Layer**: Redis integration for frequent requests
- **Load Balancing**: Multiple model instances for scaling
- **Monitoring**: Comprehensive performance and usage analytics
- **A/B Testing**: Model variant comparison framework
  

## 👤 Author

**Ayush Poddar**
- Email: ayushpoddar351@gmail.com
- LinkedIn: [Your LinkedIn Profile]
- GitHub: [Your GitHub Profile]

## 🙏 Acknowledgments

- **Hugging Face Team**: Transformers library and model hub
- **Google Research**: PEGASUS model architecture
- **FastAPI Community**: High-performance web framework
- **PyTorch Team**: Deep learning framework
- **SAMSum Dataset**: Conversation summarization data
- **ROUGE Evaluation**: Text summarization metrics

## 📚 Key Learning Outcomes

This project demonstrates:
- **Advanced NLP**: State-of-the-art transformer model fine-tuning
- **MLOps Pipeline**: Complete ML lifecycle automation
- **Production Deployment**: Scalable API service development
- **Model Evaluation**: Comprehensive performance assessment
- **Configuration Management**: YAML-based system configuration
- **Logging & Monitoring**: Production-ready observability
- **API Development**: RESTful service architecture with FastAPI

---

*This project showcases cutting-edge natural language processing capabilities using transformer models for high-quality text summarization, demonstrating modern MLOps practices and production-ready deployment strategies.*
