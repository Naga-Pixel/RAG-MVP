# ✅ Oku — Deployment Protocol (Authoritative)

**Last updated:** Feb 2026  
**Scope:** Single VPS running two isolated environments (Dev + Beta) via Docker Compose.  
**Repo on VPS:** `/home/ubuntu/oku-rag`

---

## 0) Ground Rules (Non-negotiable)

1) **Single active branch: `fts-phase-a`**
- All work lands on this branch.
- No parallel deploy branches.

2) **GitHub is the source of truth**
- Code is committed + pushed from local (laptop / Claude Code).
- VPS is **pull + run only**.
- Never commit directly on the VPS (prevents drift).

3) **Docker Compose only**
- Never use `docker run`.
- Never edit files inside containers.

4) **No `container_name:` in compose**
- Isolation is handled via project names (`-p oku-dev`, `-p oku-beta`).
- Hard-coded names cause collisions and “ghost bugs”.

5) **VPS is pull-only (push disabled)**
- Pushing to the VPS remote is forbidden and structurally blocked.

---

## 1) Remotes (Standard Naming)

**Local repo remotes must be:**

- `origin` → GitHub (fetch + push)
- `vps` → VPS (fetch only; push **DISABLED**)

Expected `git remote -v`:

```
origin  git@github.com:Naga-Pixel/RAG-MVP.git (fetch)
origin  git@github.com:Naga-Pixel/RAG-MVP.git (push)
vps     ssh://ubuntu@<VPS_IP>/home/ubuntu/oku-rag (fetch)
vps     DISABLED (push)
```

**Invariant:** All pushes go to GitHub only:

```bash
git push origin fts-phase-a
```

---

## 2) Environments (Truth Table)

| Environment | Compose Project | Compose File        | Host Port | Purpose |
|------------|-----------------|---------------------|----------:|--------|
| Dev        | `oku-dev`       | `compose.dev.yaml`  | 8001      | fast iteration |
| Beta       | `oku-beta`      | `compose.beta.yaml` | 8000      | testers |

**Key difference:**

- **Dev** mounts code (`./app:/app/app`) → restart usually enough
- **Beta** bakes code into image → rebuild required for code changes

---

## 3) Daily Workflow (Claude/Laptop → GitHub → VPS)

### A) Make a change (Local)

```bash
git checkout fts-phase-a
git pull origin fts-phase-a
# edit code
git status
git commit -am "Your change"
git push origin fts-phase-a
```

### B) Deploy to VPS (Pull + Run only)

SSH into VPS:

```bash
cd /home/ubuntu/oku-rag
git checkout fts-phase-a
git pull origin fts-phase-a
```

Then deploy to each environment:

**Dev**
```bash
sudo docker compose -p oku-dev -f compose.dev.yaml restart api
```

**Beta**
```bash
sudo docker compose -p oku-beta -f compose.beta.yaml build api --no-cache
sudo docker compose -p oku-beta -f compose.beta.yaml up -d --force-recreate api
```

---

## 4) Claude Code Workflow (Allowed vs Forbidden)

### Allowed (Local only)

Claude may:
- apply approved diffs locally
- run local compose smoke checks
- commit + push to `fts-phase-a` on GitHub

### Forbidden

Claude must never:
- SSH into VPS
- push to `vps` remote
- deploy on VPS
- push to `main`
- force-push

---

## 5) Phase A Verification Checklist (Postgres FTS Shadow)

### Required env

- `FTS_SHADOW_ENABLED=true` in container env

### Verify after ingest

**Supabase**
```sql
select count(*) from internal.chunks;
```

**Beta logs**
```bash
sudo docker logs oku-beta-api-1 --tail 300 | grep -i fts_shadow | tail -n 80
```

Expect:
- `fts_shadow_upsert ok`
- No `UndefinedTable` errors

---

## Mental Model

**GitHub → VPS** is a one-way pipe.

The VPS only ever changes state when you run:

```bash
git pull
```

Everything else is local or automated.
