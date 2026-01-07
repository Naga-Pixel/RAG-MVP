# b_rag

A modular RAG (Retrieval-Augmented Generation) system with multi-source document ingestion.

## Features

- **Multi-source ingestion**: Local files, Google Drive (more coming)
- **Idempotent sync**: Re-syncing won't create duplicate vectors
- **Multi-tenant ready**: Tenant ID support throughout the pipeline
- **Version tracking**: Automatic cleanup of old document versions
- **Connector state**: Tracks last sync time per source

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                            │
│   /ask  /sync/local  /sync/google-drive  /state            │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                   Connector Layer                           │
│   LocalFilesConnector    GoogleDriveConnector    (future)  │
│   Each returns: Document(content, metadata, external_id)   │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Ingestion Pipeline                         │
│   Loaders → Chunking → Embeddings → Qdrant (idempotent)    │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Storage Layer                            │
│   Qdrant (vectors)          SQLite (connector state)       │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
rag-mvp/
├── app/                    # FastAPI application
│   ├── main.py             # API endpoints
│   ├── config.py           # Settings (from .env)
│   ├── rag_service.py      # RAG query logic
│   ├── qdrant_client.py    # Qdrant connection
│   └── models.py           # Pydantic models
│
├── connectors/             # Data source connectors
│   ├── base.py             # BaseConnector abstract class
│   ├── local_files.py      # Local filesystem connector
│   ├── google_drive.py     # Google Drive connector
│   └── state_store.py      # SQLite state storage
│
├── ingest/                 # Document processing
│   ├── pipeline.py         # Main ingestion pipeline
│   ├── loaders.py          # File format parsers
│   ├── chunking.py         # Text chunking
│   ├── embedder.py         # OpenAI embeddings
│   └── ingest_cli.py       # CLI interface
│
├── static/                 # Frontend
│   └── index.html          # Web UI
│
├── data/
│   ├── raw/                # Local documents to ingest
│   └── state.db            # Connector state (auto-created)
│
├── credentials/            # Service account keys (gitignored)
├── tests/                  # Test suite
├── .env                    # Environment variables (gitignored)
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

# Optional (defaults shown)
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=documents_default
CHUNK_SIZE=800
CHUNK_OVERLAP=200
```

## Usage

### Web Interface

```bash
uvicorn app.main:app --reload
# Open http://localhost:8000
```

### CLI Ingestion

```bash
# Ingest local files
python -m ingest.ingest_cli --dir data/raw --recreate

# Ingest from Google Drive
python -m ingest.ingest_cli --source google_drive \
  --credentials credentials/service-account.json \
  --folder-id YOUR_FOLDER_ID
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/ask` | POST | Query documents |
| `/sync/local` | POST | Sync local files |
| `/sync/google-drive` | POST | Sync Google Drive |
| `/sources` | GET | List available connectors |
| `/state` | GET | List connector sync states |
| `/health` | GET | Health check |

#### Example: Query documents

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the lease term?"}'
```

#### Example: Sync local files

```bash
curl -X POST http://localhost:8000/sync/local \
  -H "Content-Type: application/json" \
  -d '{"directory": "data/raw", "recreate": false}'
```

## Google Drive Setup

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

## Supported File Types

| Extension | Library |
|-----------|---------|
| `.txt`, `.md` | Built-in |
| `.pdf` | pypdf |
| `.docx` | python-docx |
| `.xlsx` | openpyxl |

## Known Limitations

- **No incremental sync**: All documents are processed on each sync (uses idempotency to avoid duplicates)
- **No deleted file cleanup**: Files removed from source remain in Qdrant
- **Sync is synchronous**: Large syncs block the API; use CLI for bulk ingestion

## License

MIT
