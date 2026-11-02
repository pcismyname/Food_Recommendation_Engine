# AWS EC2 Deployment Guide

## Architecture

```
EC2 Instance (t2.medium or larger)
  └── Docker
        └── food-recommender container (port 8000)
              ├── FastAPI + uvicorn
              ├── evidence_sentence.csv  (~107 MB, baked into image)
              ├── topk_n10000_s42.csv    (Thai results, baked into image)
              └── 50_restaurants_all_rates.csv (baked into image)
```

No GPU required. Model runs on CPU only (inference is pre-computed; only new queries use the encoder).

---

## 1. Local test before deploying

```bash
# From the project root
docker compose up --build

# Open http://localhost:8000 — should see the UI
# Check health: curl http://localhost:8000/api/health
```

---

## 2. EC2 setup (once the professor gives you access)

### 2a. Connect to the instance

```bash
ssh -i your-key.pem ec2-user@<EC2_PUBLIC_IP>
```

### 2b. Install Docker on the EC2 instance

```bash
sudo yum update -y                          # Amazon Linux 2
sudo yum install -y docker
sudo service docker start
sudo usermod -aG docker ec2-user
# Log out and back in for group membership to take effect
```

### 2c. Transfer the project

**Option A — git clone (if the repo is on GitHub):**
```bash
git clone https://github.com/<your-repo>/Food_Recommendation_Engine.git
cd Food_Recommendation_Engine
```

**Option B — rsync from your machine:**
```bash
rsync -avz --exclude '.git' --exclude 'baselines/cache/*.npy' \
  /path/to/Food_Recommendation_Engine/ \
  ec2-user@<EC2_PUBLIC_IP>:~/Food_Recommendation_Engine/
```

### 2d. Build and run the container

```bash
cd ~/Food_Recommendation_Engine

# Build image (bakes in the CSVs)
docker build -t food-recommender:latest .

# Run (detached, restarts automatically)
docker run -d \
  --name food_recommender \
  --restart unless-stopped \
  -p 8000:8000 \
  -v model_cache:/model_cache \
  food-recommender:latest

# Follow logs
docker logs -f food_recommender
```

### 2e. Open the app

Add an **inbound security group rule** for port 8000:
- Type: Custom TCP
- Port: 8000
- Source: 0.0.0.0/0 (or your IP for restricted access)

Then open: `http://<EC2_PUBLIC_IP>:8000`

---

## 3. Optional: enable HTTPS with nginx + Let's Encrypt

If you have a domain name, add nginx as a reverse proxy:

```bash
sudo amazon-linux-extras install -y nginx1
sudo systemctl enable nginx
```

`/etc/nginx/conf.d/food_recommender.conf`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 4. Enabling live inference for new English queries

The container as shipped only handles the 100 pre-computed RIRD queries.
To enable arbitrary English queries:

1. Copy the sentence embeddings file into the image (adds ~750 MB):
   Uncomment the `COPY` line in `Dockerfile` that copies `emb_*.npy`.

2. The engine will auto-detect the file at startup and load the encoder.
   This adds ~30–60 seconds to startup time on CPU.

3. Minimum recommended instance: **t2.large** (8 GB RAM) for the embeddings + encoder.

---

## 5. Useful commands

```bash
# Check container status
docker ps

# Restart
docker restart food_recommender

# Update after code change
docker build -t food-recommender:latest . && docker restart food_recommender

# Stop
docker stop food_recommender && docker rm food_recommender
```
