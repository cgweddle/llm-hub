# LLM Hub Deployment Guide

## Overview

LLM Hub is deployed using Podman with 4 containers (PostgreSQL, backend, frontend, Caddy reverse proxy) orchestrated by `podman-compose`. Pushes to the `main` branch automatically deploy to the server via GitHub Actions.

LangFuse (observability) runs separately on the same server in its own Podman Compose stack.

## Architecture

```
Internet
    |
  Caddy (:80/:443) — auto TLS via Let's Encrypt
    |
    |--- /api/*  -->  Backend (:8000)  -->  PostgreSQL (:5432)
    |--- /*      -->  Frontend (:3000) -->  PostgreSQL (:5432)
                                                |
                      Backend  ------------> LangFuse (:3000 on host)
```

All services communicate on an internal `llmhub-net` bridge network. Only Caddy exposes ports to the host.

---

## Prerequisites

- A Linux VPS with Podman and podman-compose installed
- A domain name pointed at the VPS (A record)
- Ports 22, 80, and 443 open in the firewall
- Git installed on the server

---

## Server Setup (One-Time)

### 1. Create a deploy user

```bash
# As root:
useradd -m -s /bin/bash deploy
usermod -aG sudo deploy
loginctl enable-linger deploy   # Keeps containers running after SSH logout
```

### 2. Set up SSH key for deployments

On your **local machine**, generate a dedicated deploy key:

```bash
ssh-keygen -t ed25519 -C "github-deploy" -f ~/.ssh/llmhub_deploy
```

This creates:
- `~/.ssh/llmhub_deploy` — private key (goes into GitHub Secrets)
- `~/.ssh/llmhub_deploy.pub` — public key (goes on the server)

Add the public key to the server:

```bash
# As root on the VPS:
mkdir -p /home/deploy/.ssh
echo "<contents of llmhub_deploy.pub>" >> /home/deploy/.ssh/authorized_keys
chown -R deploy:deploy /home/deploy/.ssh
chmod 700 /home/deploy/.ssh
chmod 600 /home/deploy/.ssh/authorized_keys
```

### 3. Harden SSH

Edit `/etc/ssh/sshd_config`:

```
PasswordAuthentication no
PermitRootLogin no
```

Then restart: `sudo systemctl restart sshd`

### 4. Configure firewall

```bash
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP (Caddy cert renewal + redirect)
ufw allow 443/tcp   # HTTPS (application traffic)
ufw enable
```

### 5. Set up LangFuse

LangFuse runs in its own separate Podman Compose stack:

```bash
su - deploy
git clone https://github.com/langfuse/langfuse.git /opt/langfuse
cd /opt/langfuse
# Update secrets in docker-compose.yml as needed
podman compose up -d
```

Wait 2-3 minutes, then open `http://<your-server-ip>:3000`:
1. Create an account (first user becomes admin)
2. Create a project (e.g., "LLM Hub")
3. Go to **Settings** -> **API Keys** -> **Create API Key**
4. Save the generated **Public Key** (`pk-lf-...`) and **Secret Key** (`sk-lf-...`) — you'll need them for GitHub Secrets

---

## GitHub Secrets

Go to your GitHub repo: **Settings** -> **Secrets and variables** -> **Actions** -> **New repository secret**

Add these **repository secrets**:

| Secret | Value | Notes |
|---|---|---|
| `SSH_HOST` | Your VPS IP or hostname | e.g., `203.0.113.50` |
| `SSH_USER` | `deploy` | The user created above |
| `SSH_PRIVATE_KEY` | Full contents of `~/.ssh/llmhub_deploy` | Include the `-----BEGIN/END-----` lines |
| `POSTGRES_USER` | `llmhub` | PostgreSQL username |
| `POSTGRES_PASSWORD` | A strong random password | Only used on first DB creation |
| `PUBLIC_DOMAIN` | Your domain name | e.g., `llmhub.example.com` |
| `LANGFUSE_SECRET_KEY` | `sk-lf-...` | From LangFuse UI (step 5 above) |
| `LANGFUSE_PUBLIC_KEY` | `pk-lf-...` | From LangFuse UI (step 5 above) |
| `LANGFUSE_HOST` | `http://host.containers.internal:3000` | Literal value — Podman resolves this to the host machine |

### Notes on secrets

- **`POSTGRES_PASSWORD`**: PostgreSQL only reads this on **first startup** (when the data volume is empty). Changing the secret later won't change the DB password — you'd need to `ALTER USER` inside PostgreSQL.
- **`LANGFUSE_HOST`**: `host.containers.internal` is a built-in Podman hostname. Don't substitute anything — use it literally. It lets containers reach services on the host machine.
- **`SSH_PRIVATE_KEY`**: Paste the entire file including `-----BEGIN OPENSSH PRIVATE KEY-----` and `-----END OPENSSH PRIVATE KEY-----`.

---

## Deployment

### Automatic (CI/CD)

Every push to `main` triggers `.github/workflows/deploy.yml`:

1. GitHub Actions SSHes into the VPS
2. Pulls the latest code to `~/llm-hub`
3. Writes `deploy/.env` from GitHub Secrets (never committed to git)
4. Builds container images
5. Restarts all services with `podman-compose up -d`
6. Verifies the backend health check

### Manual (first time or debugging)

```bash
ssh deploy@<your-server>
cd ~/llm-hub/deploy

# Create .env manually (normally done by GitHub Actions)
cp .env.example .env   # if available, or write it by hand

# Build and start
podman-compose build --build-arg PUBLIC_API_BASE_URL=https://yourdomain.com/api
podman-compose up -d

# Check status
podman-compose ps
podman-compose logs -f backend
```

---

## Useful Commands

```bash
# View all container status
podman-compose ps

# View logs for a specific service
podman-compose logs -f backend
podman-compose logs -f frontend
podman-compose logs -f caddy

# Restart a single service
podman-compose restart backend

# Rebuild and restart a single service
podman-compose build backend
podman-compose up -d --force-recreate backend

# Database backup
podman exec llmhub-postgres pg_dump -U llmhub llm_hub > backup_$(date +%F).sql

# Database restore
cat backup.sql | podman exec -i llmhub-postgres psql -U llmhub llm_hub

# Check backend health
curl -sf https://yourdomain.com/api/healthz

# Shell into a container
podman exec -it llmhub-backend /bin/sh
podman exec -it llmhub-postgres psql -U llmhub -d llm_hub

# View Caddy TLS certificate status
podman logs llmhub-caddy | grep "certificate"
```

---

## Troubleshooting

**Containers won't start after reboot:**
Ensure `loginctl enable-linger deploy` was run. Without it, Podman containers stop when the deploy user's session ends.

**Backend can't reach PostgreSQL:**
Check that the postgres container is healthy: `podman-compose ps`. If it shows "starting", wait for the health check. Check logs: `podman-compose logs postgres`.

**Caddy can't get a TLS certificate:**
Make sure ports 80 and 443 are open in the firewall, and your domain's DNS A record points to the server's IP. Check: `podman logs llmhub-caddy`.

**Backend can't reach LangFuse:**
Verify LangFuse is running on the host: `curl http://localhost:3000`. If it is, the issue may be `host.containers.internal` resolution — try using the server's actual IP instead in `LANGFUSE_HOST`.

**Changed POSTGRES_PASSWORD but can't connect:**
The password is only set on first DB initialization. Either `ALTER USER` inside PostgreSQL, or delete the volume (`podman volume rm deploy_pgdata`) and reinitialize (destroys all data).

**GitHub Actions deploy fails:**
Check the Actions tab in GitHub for logs. Common issues:
- SSH key mismatch — verify the public key is in `/home/deploy/.ssh/authorized_keys`
- Podman not installed or not in PATH for the deploy user
- Disk space full on server
