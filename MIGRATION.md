# Domain Migration: oku.systems

This document outlines the steps to migrate OKU from the current domain to `https://oku.systems`.

## Overview

- **New Primary Domain:** `https://oku.systems`
- **WWW Variant:** `https://www.oku.systems` (optional, for user convenience)
- **Strategy:** Additive migration - keep old domain working during transition

---

## 1. Codebase Changes (DONE)

The following changes have been made to support the new domain:

### CORS Origins (`app/config.py`)
Added `https://oku.systems` and `https://www.oku.systems` to the default CORS origins list.

**Before:**
```python
default=["http://localhost:3000", "http://localhost:8000"]
```

**After:**
```python
default=[
    "http://localhost:3000",
    "http://localhost:8000",
    "https://oku.systems",
    "https://www.oku.systems",
]
```

### Diagnostic Logging
Added logging to help debug cutover issues:
- `/ask` endpoint logs `origin` and `host` headers
- OAuth start logs `redirect_uri`
- OAuth callback logs `postMessage_origin`

---

## 2. Environment Variables (VPS)

Update `.env` on the VPS to include the new domain in CORS and set the OAuth redirect URI:

```bash
# Add to existing CORS_ORIGINS (comma-separated, no spaces)
CORS_ORIGINS=http://localhost:3000,http://localhost:8000,https://oku.systems,https://www.oku.systems

# Update Google Drive OAuth redirect URI to new domain
GOOGLE_DRIVE_REDIRECT_URI=https://oku.systems/oauth/google/drive/callback
```

**Note:** If the old domain should continue working for Google Drive OAuth during transition, you may need to keep both redirect URIs configured in Google Cloud Console (see below).

---

## 3. Supabase Console Checklist

Go to: **Supabase Dashboard > Project > Authentication > URL Configuration**

### Site URL
```
https://oku.systems
```

### Redirect URLs (add all of these)
```
https://oku.systems
https://oku.systems/**
https://www.oku.systems
https://www.oku.systems/**
```

**Keep existing entries** (e.g., `http://localhost:*`, old domain) until migration is complete.

### Verification
After updating:
1. Test Google OAuth login from `https://oku.systems`
2. Verify redirect lands back on `https://oku.systems` (not old domain)
3. Check browser console for any auth errors

---

## 4. Google Cloud Console Checklist

Go to: **Google Cloud Console > APIs & Services > Credentials > OAuth 2.0 Client IDs > [Your Client]**

### Authorized JavaScript Origins

Add:
```
https://oku.systems
https://www.oku.systems
```

Keep existing:
```
http://localhost:8000
http://localhost:8001
[old domain if any]
```

### Authorized Redirect URIs

Add:
```
https://oku.systems/oauth/google/drive/callback
```

**Note:** The Supabase Google OAuth callback URL (e.g., `https://[project-ref].supabase.co/auth/v1/callback`) does NOT need to change - it's handled by Supabase.

Keep existing:
```
http://localhost:8000/oauth/google/drive/callback
[old domain callback if any]
```

### Verification
After updating:
1. Test "Connect Google Drive" from `https://oku.systems`
2. Verify OAuth popup completes successfully
3. Check that folder selection works via Google Picker

---

## 5. DNS Configuration

Configure DNS records for `oku.systems`:

```
# A record pointing to VPS IP
oku.systems.     A     [VPS_IP_ADDRESS]

# Optional: WWW redirect or same A record
www.oku.systems. A     [VPS_IP_ADDRESS]
# OR
www.oku.systems. CNAME oku.systems.
```

---

## 6. SSL/TLS Certificate

Ensure SSL certificate covers both domains:
- `oku.systems`
- `www.oku.systems` (if using)

If using Let's Encrypt with certbot:
```bash
certbot certonly --nginx -d oku.systems -d www.oku.systems
```

---

## 7. Nginx Configuration (if applicable)

Add server block for new domain:

```nginx
server {
    listen 443 ssl http2;
    server_name oku.systems www.oku.systems;

    ssl_certificate /etc/letsencrypt/live/oku.systems/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/oku.systems/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 8. Step Order for Cutover

1. **DNS propagation** - Add DNS records (can take up to 48h, usually faster)
2. **SSL certificate** - Generate cert for new domain
3. **Nginx config** - Add server block for new domain
4. **Supabase config** - Add new redirect URLs
5. **Google Cloud config** - Add new JavaScript origins and redirect URIs
6. **VPS .env** - Update `GOOGLE_DRIVE_REDIRECT_URI` to new domain
7. **Restart app** - `docker compose restart api`
8. **Test** - Verify all flows work on new domain
9. **Monitor** - Watch logs for origin/CORS issues

---

## 9. Rollback Plan

If issues occur after cutover:

### Quick Rollback (revert to old domain)
1. Revert `GOOGLE_DRIVE_REDIRECT_URI` in `.env` to old domain
2. Restart app: `docker compose restart api`
3. Users continue using old domain

### Partial Rollback (keep new domain, fix specific issues)
1. Check logs: `docker compose logs api | grep -E '\[ask\]|\[drive_oauth\]'`
2. Verify CORS origins include the problematic origin
3. Verify Google Cloud Console has the correct redirect URIs
4. Verify Supabase has the correct redirect URLs

### Log Analysis
Look for these patterns in logs:
- `[ask] origin=https://oku.systems` - confirms requests coming from new domain
- `[drive_oauth_start] redirect_uri=` - shows configured redirect
- `[drive_oauth_callback] postMessage_origin=` - shows callback origin

---

## 10. Post-Migration Cleanup (After Transition Period)

Once the new domain is stable (e.g., 2-4 weeks):

1. **Optional:** Remove old domain from CORS origins
2. **Optional:** Remove old domain from Supabase redirect URLs
3. **Optional:** Remove old redirect URI from Google Cloud Console
4. **Recommended:** Set up redirect from old domain to new domain

---

## Quick Reference

| Setting | Old Value | New Value |
|---------|-----------|-----------|
| CORS_ORIGINS | localhost only | + `https://oku.systems`, `https://www.oku.systems` |
| GOOGLE_DRIVE_REDIRECT_URI | `[old]/oauth/google/drive/callback` | `https://oku.systems/oauth/google/drive/callback` |
| Supabase Site URL | [old] | `https://oku.systems` |
| Google JS Origins | [old] | + `https://oku.systems`, `https://www.oku.systems` |
| Google Redirect URIs | [old] | + `https://oku.systems/oauth/google/drive/callback` |
