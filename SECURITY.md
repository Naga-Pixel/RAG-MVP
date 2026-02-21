# Security Overview

Last audited: 2026-02-21

## Security Fixes Applied

| Issue | Severity | Fix | Commit |
|-------|----------|-----|--------|
| SQL Injection in FTS queries | Critical | `psycopg2.sql.SQL` + `sql.Identifier` for all table/schema names | `aed8ad7` |
| JWT Algorithm Confusion | High | Restricted to expected algorithms only (`HS256`, `ES256`) | `c208d28` |
| Fernet Key Validation | High | Validate encryption key at startup, fail fast on invalid | `3eabc0e` |

## Data Flow: What Goes to OpenAI

| When | API | Data Sent |
|------|-----|-----------|
| Document ingestion | Embeddings | Chunk text (~800 chars each) |
| User query | Embeddings | Question text only |
| Query rewrite | Chat (gpt-4o-mini) | Question + last 3 conversation turns |
| Answer generation | Chat | System prompt + context chunks + question |
| Document summary (if enabled) | Chat | Full document text |
| Voice transcription (if enabled) | Whisper | Audio file |

### What OpenAI NEVER Sees

- Full PDF/file uploads (only extracted text chunks)
- File paths or system paths
- Tenant IDs (filtered locally)
- Database credentials
- Your Qdrant/Postgres data directly

### OpenAI Data Usage

- **API terms explicitly prohibit training on API data**
- This is different from ChatGPT consumer products
- See: https://openai.com/policies/api-data-usage-policies

## Infrastructure Security

### Qdrant (Vector Database)

**Status: SECURE**

- Runs in Docker with internal networking only
- Not exposed to public internet (no port mapping to `0.0.0.0`)
- Only accessible by app containers via `http://qdrant:6333`

Verified with:
```bash
sudo netstat -tlnp | grep 6333          # No output = not publicly listening
sudo docker ps | grep qdrant            # Shows 6333-6334/tcp (internal only)
curl http://SERVER_IP:6333/collections  # Timeout = not accessible
```

### Postgres (FTS + OAuth Data)

- Using external managed service (Supabase)
- Security managed by provider
- Ensure `DATABASE_URL` uses SSL (`?sslmode=require`)

## Comparison: Oku vs ChatGPT

| Aspect | Oku | ChatGPT |
|--------|-----|---------|
| File storage | Your Qdrant (you control) | OpenAI servers |
| Data sent | Text chunks via API | Full files |
| Training on data | No (API terms) | Depends on plan/settings |
| Security certifications | None (self-hosted) | SOC 2 Type 2 |

**Bottom line:** For data privacy/control, Oku is better. For enterprise security maturity, ChatGPT Enterprise has more resources.

## Remaining Security Items

### High Priority

- [x] **File upload content validation** - Validate magic bytes, not just file extension (see `app/file_validation.py`)
- [x] **Pin dependencies** - All dependencies pinned in `requirements.txt`

### Medium Priority

- [ ] Rate limiting per tenant (currently global)
- [ ] Audit logging for sensitive operations
- [ ] Add security headers (CSP, HSTS) if not handled by reverse proxy

## Configuration Checklist

Production deployment should have:

```bash
# Required
QDRANT_API_KEY=<set if Qdrant exposed>    # Not needed if Docker-internal
DATABASE_URL=postgres://...?sslmode=require
DRIVE_TOKEN_ENCRYPTION_KEY=<valid Fernet key>
SUPABASE_JWT_SECRET=<from Supabase dashboard>
API_KEY=<for admin endpoints>

# Recommended
CORS_ORIGINS=https://yourdomain.com       # Not wildcard
RATE_LIMIT_ASK=10/minute
RATE_LIMIT_SYNC=5/minute
```

## Verifying Security

### Test Qdrant is not exposed
```bash
# From outside your server
curl -s --max-time 3 http://YOUR_SERVER_IP:6333/collections
# Should timeout or refuse (not return data)
```

### Test tenant isolation
```bash
# Create test data in tenant A, query from tenant B
# Should NOT see tenant A's documents
```

### Generate Fernet key
```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```
