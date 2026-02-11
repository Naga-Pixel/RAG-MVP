# Oku

A modular RAG (Retrieval-Augmented Generation) system with multi-source document ingestion, hybrid retrieval, and OCR support.

## Features

### Core
- **Multi-source ingestion**: Local files, Google Drive, browser uploads
- **Idempotent sync**: Re-syncing won't create duplicate vectors
- **Multi-tenant ready**: Tenant ID support throughout the pipeline
- **Incremental sync**: Only processes files modified since last sync (Google Drive)

### Retrieval
- **Hybrid search**: Combines vector and keyword retrieval with RRF fusion
- **Reranking**: Cross-encoder reranking for improved relevance
- **Scope resolution**: Auto-detects document/folder references in queries
- **Conversational context**: Rewrites follow-up questions into standalone queries

### Document Processing
- **OCR support**: Google Vision API for scanned PDFs and images
- **CSV ingestion**: Deterministic text extraction with row-based formatting
- **RTF support**: Rich Text Format parsing
- **Audio transcription**: Whisper-powered push-to-talk input

### Security & Operations
- **Supabase authentication**: JWT-based user authentication
- **API key protection**: Secure admin/sync endpoints
- **Rate limiting**: Configurable per-endpoint limits
- **CORS configuration**: Customizable allowed origins
- **Sentry integration**: Error tracking and monitoring

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
│   /ask  /sync/upload  /sync/google-drive  /folders /docs   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    RAG Pipeline                             │
│   Query Rewrite → Hybrid Retrieve → Rerank → Generate      │
│   (context-aware)  (vector + FTS)   (cross-encoder)        │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   Connector Layer                           │
│   LocalFilesConnector    GoogleDriveConnector    Upload    │
│   Each returns: Document(content, metadata, external_id)   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Ingestion Pipeline                         │
│   Loaders → OCR (optional) → Chunking → Embeddings → Store │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Storage Layer                            │
│   Qdrant (vectors)    Postgres (FTS + OAuth)    SQLite     │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
b_rag/
├── app/                      # FastAPI application
│   ├── main.py               # API endpoints
│   ├── config.py             # Settings (from .env)
│   ├── rag_service.py        # RAG query logic + query rewriting
│   ├── qdrant_client.py      # Qdrant connection
│   ├── models.py             # Pydantic models
│   ├── reranker.py           # Cross-encoder reranking
│   ├── hybrid.py             # RRF fusion for hybrid search
│   ├── keyword_retrieval.py  # FTS keyword search
│   ├── scope_resolver.py     # Auto-scope detection
│   ├── google_drive_auth.py  # OAuth + Picker integration
│   ├── transcription.py      # Whisper audio transcription
│   └── logging_config.py     # Structured logging
│
├── connectors/               # Data source connectors
│   ├── base.py               # BaseConnector abstract class
│   ├── local_files.py        # Local filesystem connector
│   ├── google_drive.py       # Google Drive connector
│   └── state_store.py        # SQLite state storage
│
├── ingest/                   # Document processing
│   ├── pipeline.py           # Main ingestion pipeline
│   ├── loaders.py            # File format parsers (PDF, DOCX, XLSX, RTF, CSV, images)
│   ├── chunking.py           # Text chunking
│   ├── embedder.py           # OpenAI embeddings
│   ├── ocr.py                # Google Vision OCR
│   ├── csv_loader.py         # CSV to text conversion
│   └── ingest_cli.py         # CLI interface
│
├── static/                   # Frontend
│   └── index.html            # Web UI with Drive picker, autocomplete
│
├── migrations/               # SQL migrations
├── credentials/              # Service account keys (gitignored)
├── tests/                    # Test suite
├── .env                      # Environment variables (gitignored)
├── Dockerfile                # Container build
└── requirements.txt
```

## Setup

### Prerequisites

- Python 3.11+
- Docker (for Qdrant)
- OpenAI API key

### Installation

```bash
# Clone the repo
git clone https://github.com/Naga-Pixel/RAG-MVP.git
cd RAG-MVP

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant
```

### Configuration

Edit `.env`:

```bash
# Required
OPENAI_API_KEY=sk-...

# Qdrant (vector store)
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents_default

# Ingestion
CHUNK_SIZE=800
CHUNK_OVERLAP=200

# Retrieval
RETRIEVE_K=12                    # Vector candidates to retrieve
FINAL_K=6                        # Chunks in final context
RERANK_ENABLED=false             # Enable cross-encoder reranking
MAX_CHUNKS_PER_DOC=3             # Diversity: max chunks per document

# Hybrid Search (requires Postgres FTS)
HYBRID_ENABLED=false             # Enable RRF fusion
HYBRID_VEC_K=12                  # Vector candidates for fusion
HYBRID_KW_K=10                   # Keyword candidates for fusion
HYBRID_RRF_K=60                  # RRF constant

# OCR (for scanned PDFs and images)
OCR_ENABLED=false
OCR_PROVIDER=google_vision
OCR_MIN_TEXT_LENGTH=300          # PDFs with less text trigger OCR
OCR_GOOGLE_CREDENTIALS_PATH=credentials/vision-sa.json

# Authentication
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_ANON_KEY=...
SUPABASE_JWT_SECRET=...

# Security
API_KEY=...                      # Protects sync endpoints
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
RATE_LIMIT_ASK=10/minute
RATE_LIMIT_SYNC=5/minute

# Google Drive OAuth (for browser picker)
GOOGLE_DRIVE_CLIENT_ID=...
GOOGLE_DRIVE_CLIENT_SECRET=...
GOOGLE_PICKER_API_KEY=...
DRIVE_TOKEN_ENCRYPTION_KEY=...   # Fernet key

# Postgres (for FTS + OAuth tokens)
DATABASE_URL=postgresql://...

# Monitoring
SENTRY_DSN=...
SENTRY_ENVIRONMENT=production
```

## Usage

### Web Interface

```bash
uvicorn app.main:app --reload
# Open http://localhost:8000
```

Features:
- Document search with autocomplete
- Folder/document scoping
- Conversation history for follow-up questions
- Push-to-talk audio input
- Google Drive folder picker (with OAuth)
- Local file upload

### CLI Ingestion

```bash
# Ingest local files
python -m ingest.ingest_cli --dir data/raw --recreate

# Ingest from Google Drive
python -m ingest.ingest_cli --source google_drive \
  --credentials credentials/service-account.json \
  --folder-id YOUR_FOLDER_ID

# Ingest with OCR enabled (set OCR_ENABLED=true in .env)
python -m ingest.ingest_cli --dir data/scanned-docs
```

### API Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | - | Web UI |
| `/ask` | POST | Supabase JWT | Query documents with RAG |
| `/folders` | GET | Supabase JWT | List user's synced folders |
| `/documents` | GET | Supabase JWT | List documents (optionally by folder) |
| `/sync/upload` | POST | Supabase JWT | Upload files from browser |
| `/sync/local` | POST | API Key | Sync local directory |
| `/sync/google-drive` | POST | API Key | Sync Google Drive (service account) |
| `/transcribe` | POST | Supabase JWT | Transcribe audio (Whisper) |
| `/sources` | GET | - | List available connectors |
| `/state` | GET | API Key | List connector sync states |
| `/health` | GET | - | Health check |
| `/config/frontend` | GET | - | Frontend config (Supabase, Picker keys) |
| `/oauth/google/drive/*` | - | - | OAuth flow for Drive picker |

#### Example: Query documents

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <supabase_jwt>" \
  -d '{
    "query": "What is the lease term?",
    "folder_id": "abc123",
    "conversation_history": [
      {"role": "user", "content": "Tell me about the contract"},
      {"role": "assistant", "content": "The contract covers..."}
    ]
  }'
```

#### Example: Upload files (browser)

```bash
curl -X POST http://localhost:8000/sync/upload \
  -H "Authorization: Bearer <supabase_jwt>" \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx" \
  -F "folder_name=My Project"
```

#### Example: Sync local files (admin)

```bash
curl -X POST http://localhost:8000/sync/local \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"directory": "data/raw", "recreate": false}'
```

## Google Drive Setup

### Option 1: Service Account (CLI/API sync)

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a project and enable **Google Drive API**
3. Create a **Service Account** under Credentials
4. Download the JSON key to `credentials/service-account.json`
5. Share your Drive folder with the service account email
6. Get the folder ID from the Drive URL

```bash
python -m ingest.ingest_cli --source google_drive \
  --credentials credentials/service-account.json \
  --folder-id 1ABC...xyz
```

### Option 2: OAuth + Picker (Browser UI)

For end-users to select files via the Drive Picker:

1. In Google Cloud Console, enable **Google Drive API** and **Google Picker API**
2. Create **OAuth 2.0 Client ID** (Web application type)
3. Add authorized redirect URI: `https://yourdomain.com/oauth/google/drive/callback`
4. Create an **API Key** for the Picker (restrict to Picker API)
5. Generate a Fernet key for token encryption:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

6. Configure in `.env`:

```bash
GOOGLE_DRIVE_CLIENT_ID=...
GOOGLE_DRIVE_CLIENT_SECRET=...
GOOGLE_DRIVE_REDIRECT_URI=https://yourdomain.com/oauth/google/drive/callback
GOOGLE_PICKER_API_KEY=...
DRIVE_TOKEN_ENCRYPTION_KEY=...
```

## Adding a New Connector

1. Create `connectors/my_source.py`:

```python
from connectors.base import BaseConnector, Document, SourceType

class MySourceConnector(BaseConnector):
    
    @property
    def source_type(self) -> SourceType:
        return SourceType.API  # or add new type
    
    @property
    def name(self) -> str:
        return "My Source"
    
    def connect(self) -> bool:
        # Establish connection
        return True
    
    def disconnect(self) -> None:
        # Cleanup
        pass
    
    def list_documents(self) -> list[dict]:
        # Return list of {source_id, title, ...}
        return []
    
    def fetch_document(self, source_id: str) -> Document | None:
        # Fetch and return Document
        return Document(
            content="...",
            source_type=self.source_type,
            source_id=source_id,
            external_id=source_id,  # Stable ID for idempotency
            title="...",
        )
```

2. Register in `connectors/__init__.py`
3. Add CLI option in `ingest/ingest_cli.py`
4. Add API endpoint in `app/main.py`

## Idempotent Sync

The system generates deterministic point IDs from:
- `tenant_id`
- `source_type`
- `external_id` (stable document ID)
- `updated_at` (version marker)
- `chunk_index`

This means:
- Re-syncing same content → same IDs → upsert overwrites (no duplicates)
- Updated content → new `updated_at` → new IDs + old version cleanup

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run idempotency tests (requires Qdrant + OpenAI)
pytest tests/test_idempotency.py -v -s
```

## Docker

```bash
# Build image (includes OCR dependencies)
docker build -t b_rag .

# Run with environment variables
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e QDRANT_URL=http://host.docker.internal:6333 \
  b_rag
```

## Supported File Types

| Extension | Library | Notes |
|-----------|---------|-------|
| `.txt`, `.md` | Built-in | UTF-8 encoded |
| `.pdf` | pypdf | OCR fallback for scanned docs |
| `.docx` | python-docx | Paragraph text only |
| `.xlsx` | openpyxl | All sheets, pivot tables |
| `.rtf` | striprtf | Rich Text Format |
| `.csv` | Built-in | Row-based text format |
| `.jpg`, `.png`, `.tiff`, `.bmp`, `.gif`, `.webp` | google-cloud-vision | Requires `OCR_ENABLED=true` |

## OCR Setup

OCR enables processing of scanned PDFs and images. Uses Google Cloud Vision API.

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Enable the **Cloud Vision API**
3. Create a **Service Account** with Vision API access
4. Download the JSON key to `credentials/vision-sa.json`
5. Configure in `.env`:

```bash
OCR_ENABLED=true
OCR_PROVIDER=google_vision
OCR_GOOGLE_CREDENTIALS_PATH=credentials/vision-sa.json
```

For scanned PDF processing, also install poppler:
```bash
# macOS
brew install poppler

# Ubuntu/Debian
apt-get install poppler-utils
```

## Hybrid Search Setup

Hybrid search combines vector similarity with keyword matching using RRF fusion. Requires Postgres with FTS.

1. Set up Postgres (e.g., via Supabase)
2. Run the FTS migration in `migrations/`
3. Enable during ingestion:

```bash
FTS_SHADOW_ENABLED=true
DATABASE_URL=postgresql://...
```

4. Enable at query time:

```bash
HYBRID_ENABLED=true
```

## Known Limitations

- **Sync is synchronous**: Large syncs block the API; use CLI for bulk ingestion
- **OCR costs**: Google Vision API charges per image processed

## License

MIT
