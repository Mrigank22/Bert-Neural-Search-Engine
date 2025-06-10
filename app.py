from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse,FileResponse
from google.cloud import storage
from google.auth import exceptions
from google.oauth2 import service_account
import tempfile
import shutil
from pydantic import BaseModel
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
from typing import List, Dict
import uvicorn


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
nltk_data_dir = os.path.join(BASE_DIR, 'nltk_data')
nltk.data.path.append(nltk_data_dir)
# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
    print("Found punkt_tab tokenizer")
except LookupError:
    print("did not find punkt_tab")
    try:
        nltk.data.find('tokenizers/punkt')
        print("Found punkt tokenizer")
    except LookupError:
        print("did not find punkt")
    
try:
    nltk.data.find('corpora/stopwords')
    print("Found stopwords corpus")
except LookupError:
    print("did not find stopwords corpus")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BertRanker(nn.Module):
    """BERT-based neural reranker model."""
    def __init__(self, bert_model_name="bert-base-uncased"):
        super(BertRanker, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class SearchQuery(BaseModel):
    query: str
    top_k: int = 10

class SearchResult(BaseModel):
    pid: str
    passage: str
    bm25_score: float
    neural_score: float
    rank: int

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_time: float

# Initialize FastAPI app
app = FastAPI(title="Neural Search Engine", description="BERT-based search engine with BM25 retrieval")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (for serving the frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for model and data
model = None
tokenizer = None
bm25 = None
passages_dict = None
idx_to_pid = None

def tokenize_text(text):
    """Tokenize text for BM25 indexing."""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
    return tokens

def search_passages(query: str, top_k: int = 10, rerank_top_k: int = 100):
    """Search function that combines BM25 retrieval with neural reranking."""
    import time
    start_time = time.time()
    
    # First stage: BM25 retrieval
    tokenized_query = tokenize_text(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(bm25_scores)[::-1][:rerank_top_k]
    
    candidates = []
    for idx in top_indices:
        pid = idx_to_pid[idx]
        passage = passages_dict[pid]
        score = bm25_scores[idx]
        candidates.append({
            'pid': pid,
            'passage': passage,
            'bm25_score': float(score)
        })
    
    # Second stage: Neural reranking
    model.eval()
    rerank_scores = []
    
    for candidate in candidates:
        # Tokenize query and passage pair
        encoding = tokenizer.encode_plus(
            query,
            candidate['passage'],
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        token_type_ids = encoding['token_type_ids'].to(device)
        
        with torch.no_grad():
            score = model(input_ids, attention_mask, token_type_ids).item()
        
        rerank_scores.append(score)
    
    # Sort by neural score and return top results
    for i, score in enumerate(rerank_scores):
        candidates[i]['neural_score'] = float(score)
    
    candidates.sort(key=lambda x: x['neural_score'], reverse=True)
    
    # Format results
    results = []
    for i, candidate in enumerate(candidates[:top_k]):
        results.append(SearchResult(
            pid=candidate['pid'],
            passage=candidate['passage'],
            bm25_score=candidate['bm25_score'],
            neural_score=candidate['neural_score'],
            rank=i + 1
        ))
    
    total_time = time.time() - start_time
    return SearchResponse(query=query, results=results, total_time=total_time)

@app.on_event("startup")
async def load_model_and_data():
    global model, tokenizer, bm25, passages_dict, idx_to_pid
    
    print("Loading model and data from Google Cloud Storage...")
    try:
        # Try Application Default Credentials (ADC)
        credentials, project = google.auth.default()
    except exceptions.DefaultCredentialsError:
        # Fallback to service account key
        credentials = service_account.Credentials.from_service_account_file(
            os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        )
    
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket("search-engine-model-1")

    # Create temporary directories
    model_temp_dir = tempfile.mkdtemp()
    tokenizer_temp_dir = tempfile.mkdtemp()

    try:
        # Download model file
        model_blob = bucket.blob("best_model_all.pt")
        model_path = f"{model_temp_dir}/best_model_all.pt"
        model_blob.download_to_filename(model_path)

        # Download search data
        search_data_blob = bucket.blob("search_data.pkl")
        search_data_path = f"{model_temp_dir}/search_data.pkl"
        search_data_blob.download_to_filename(search_data_path)

        # Download tokenizer files
        tokenizer_files = [
            "neural_search_model/special_tokens_map.json",
            "neural_search_model/tokenizer_config.json",
            "neural_search_model/vocab.txt"
        ]
        
        for file in tokenizer_files:
            blob = bucket.blob(file)
            blob.download_to_filename(f"{tokenizer_temp_dir}/{file.split('/')[-1]}")

        # Load tokenizer from temp directory
        tokenizer = BertTokenizer.from_pretrained(tokenizer_temp_dir)

        # Load model
        model = BertRanker()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Load search data
        with open(search_data_path, 'rb') as f:
            search_data = pickle.load(f)
            passages_dict = search_data['passages_dict']
            idx_to_pid = search_data['idx_to_pid']
            bm25 = search_data['bm25']

        print(f"Loaded {len(passages_dict)} passages")
        print("Model and data loaded successfully from GCS!")

    finally:
        # Cleanup temporary directories
        shutil.rmtree(model_temp_dir)
        shutil.rmtree(tokenizer_temp_dir)

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(search_query: SearchQuery):
    """Search endpoint that returns ranked results."""
    if not search_query.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        results = search_passages(search_query.query, search_query.top_k)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/stats")
async def get_stats():
    """Get statistics about the search engine."""
    if passages_dict is None:
        return {"error": "Data not loaded"}
    
    return {
        "total_passages": len(passages_dict),
        "model_device": str(device),
        "model_type": "BERT-based neural reranker"
    }

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page."""
    try:
        return FileResponse(
            path="static/index.html",
            media_type="text/html; charset=utf-8"  # Add charset here
        )
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend not found</h1>",
            status_code=404
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
