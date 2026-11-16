# Deployment on AWS (AWS Academy Learner Lab)

This is how the demo app was deployed on **2026-11-17**. It covers what was
created, why, and how to start, update and shut it down.

**Live URL (only while a lab session is running):** http://3.227.65.215

---

## 1. Architecture

```
Internet ──HTTP :80──▶ Elastic IP 3.227.65.215
                          │
                 EC2 t3.large (Amazon Linux 2023, us-east-1)
                 2 vCPU · 8 GB RAM · 30 GB gp3 · 2 GB swap
                          │
                 Docker container "food_recommender"  (host :80 → container :8000)
                   ├─ FastAPI + uvicorn (app/main.py)
                   ├─ baked into image: app/, Thai CSVs, evidence_sentence.csv,
                   │                    50_restaurants_all_rates.csv
                   ├─ mounted read-only: baselines/cache (740 MB sentence embeddings
                   │                     → enables live English queries), datasets/, .env (Groq key)
                   └─ volume model_cache: Hugging Face models (downloaded on first start)
```

| Resource | Name / ID | Why |
|---|---|---|
| EC2 instance | `food-recommender` / `i-0b181282806c5c6f6` | runs the container |
| Instance type | `t3.large` (8 GB) | 740 MB embeddings + bi-encoder + cross-encoder + pandas data need ~4 GB; `t3.medium` (4 GB) would be too tight |
| Disk | 30 GB gp3 | Docker image (~2.5 GB with CPU-only torch) + data + model cache |
| Key pair | `food-rec-key` → `~/.ssh/food-rec-key.pem` (local only) | SSH access |
| Security group | `food-rec-sg` / `sg-0ff69d8332a961c6a` | port 22 only from the deployer's IP, port 80 from anywhere |
| Elastic IP | `3.227.65.215` / `eipalloc-0fb4be8fb479bb001` | the address stays the same across lab sessions |

### Learner Lab constraints that shaped this
- Only `us-east-1` / `us-west-2` are allowed, so `us-east-1` was used.
- Credentials are **temporary**: a new access key, secret and session token every session.
  Copy them from *AWS Details → AWS CLI → Show* into `~/.aws/credentials`.
- **Sessions last 4 hours.** When a session ends, the EC2 instance is **stopped**
  (not deleted). When the next session starts, the lab restarts it, and the
  container comes back automatically (`--restart unless-stopped`). **The app is only
  online while a lab session is running.**
- Budget is $50 (see the cost section).

---

## 2. What was done, step by step

All commands were run from Git Bash on Windows with the AWS CLI v2.

> Git Bash gotcha: it rewrites arguments like `/dev/xvda` into Windows paths.
> Set `MSYS_NO_PATHCONV=1` before AWS CLI calls that contain `/...` arguments.

### 2.1 Credentials
```bash
# paste the [default] block from AWS Details → AWS CLI into ~/.aws/credentials
printf "[default]\nregion = us-east-1\noutput = json\n" > ~/.aws/config
aws sts get-caller-identity          # must print an assumed-role/voclabs ARN
```

### 2.2 Key pair, security group, AMI
```bash
export AWS_DEFAULT_REGION=us-east-1 MSYS_NO_PATHCONV=1
aws ec2 create-key-pair --key-name food-rec-key --key-type rsa \
  --query KeyMaterial --output text > ~/.ssh/food-rec-key.pem
chmod 600 ~/.ssh/food-rec-key.pem

MYIP=$(curl -s https://checkip.amazonaws.com)
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 create-security-group --group-name food-rec-sg \
      --description "Food recommender web + SSH" --vpc-id $VPC --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id $SG --ip-permissions \
  "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=$MYIP/32}]" \
  "IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges=[{CidrIp=0.0.0.0/0}]"

# Latest Amazon Linux 2023 AMI (the SSM public parameter is blocked in Learner Lab)
aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=al2023-ami-2023*-kernel-*-x86_64" "Name=state,Values=available" \
  --query 'sort_by(Images,&CreationDate)[-1].ImageId' --output text
# → ami-0b71756f4d851f798
```

### 2.3 Launch the instance
User data (runs once at first boot) installs Docker and git, and adds 2 GB swap:
```bash
#!/bin/bash
dnf install -y docker git
systemctl enable --now docker
usermod -aG docker ec2-user
fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
echo '/swapfile swap swap defaults 0 0' >> /etc/fstab
```
```bash
aws ec2 run-instances --image-id ami-0b71756f4d851f798 --instance-type t3.large \
  --key-name food-rec-key --security-group-ids sg-0ff69d8332a961c6a \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=30,VolumeType=gp3,DeleteOnTermination=true}' \
  --user-data file://userdata.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=food-recommender}]'

aws ec2 allocate-address --domain vpc                       # → 3.227.65.215
aws ec2 associate-address --instance-id i-0b181282806c5c6f6 \
  --allocation-id eipalloc-0fb4be8fb479bb001
```

### 2.4 Code and data
The code comes from GitHub. The large files are git-ignored, so they were copied with `scp`:
```bash
ssh -i ~/.ssh/food-rec-key.pem ec2-user@3.227.65.215 \
  'git clone https://github.com/pcismyname/Food_Recommendation_Engine.git'

K="-i ~/.ssh/food-rec-key.pem"; R=ec2-user@3.227.65.215:Food_Recommendation_Engine
scp $K Dockerfile .env $R/
scp $K datasets/rir_data-main/reviews/50_restaurants_all_rates.csv  $R/datasets/rir_data-main/reviews/
scp $K baselines/cache/evidence_sentence.csv                        $R/baselines/cache/
scp $K baselines/cache/sentence/emb_paraphrase-multilingual-mpnet-base-v2_sentence_len256.npy \
                                                                    $R/baselines/cache/sentence/
```

### 2.5 Dockerfile change
`sentence-transformers` pulls the default PyPI `torch` wheel, which bundles
about 4 GB of CUDA libraries that a CPU-only EC2 instance never uses. The
builder stage now installs the CPU-only wheel:
```dockerfile
RUN pip install --no-cache-dir --prefix=/install \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      torch -r requirements.txt
```

### 2.6 Build and run (on the EC2 instance)
```bash
ssh -i ~/.ssh/food-rec-key.pem ec2-user@3.227.65.215
cd Food_Recommendation_Engine
docker build -t food-recommender:latest .

docker run -d --name food_recommender --restart unless-stopped \
  -p 80:8000 \
  -v "$PWD/baselines/cache:/app_root/baselines/cache:ro" \
  -v "$PWD/datasets:/app_root/datasets:ro" \
  -v "$PWD/.env:/app_root/.env:ro" \
  -v model_cache:/model_cache \
  food-recommender:latest

docker logs -f food_recommender      # wait for startup to finish
```
`.env` is mounted as a file rather than passed with `--env-file`, because it
contains `GROQ_API_KEY =...` (a space before `=`), which Docker's env-file
parser rejects. `app/retrieval.py` reads `.env` itself and strips the space.

### 2.7 Verify
```bash
curl http://3.227.65.215/api/health
```
Then open http://3.227.65.215 in a browser.

Results on 2026-11-17 (tested from outside AWS):

| Check | Result |
|---|---|
| Docker image size | 1.74 GB (CPU-only torch) |
| Startup log | `Engine ready \| EN pre-computed: 100 \| TH pre-computed: 24 \| Live inference: yes (bi-encoder + cross-encoder) \| Query rewriting: yes (Groq)` |
| RAM in use after startup | 2.1 GB of 7.6 GB |
| `GET /` | 200 |
| `GET /api/health` | `{"status":"ok","engine_ready":true}` |
| `GET /api/queries` | 100 English queries |
| `POST /api/query` live English *"quiet place with spicy noodles for a late dinner"* | rewritten by Groq to *"quiet late night spicy noodles dim sum"*; top hit GB Hand-pulled Noodles; 2.9 s |
| `POST /api/query` Thai query from `thai/thai_queries.csv` | pre-computed hit, BM25/dense sentences returned |

---

## 3. Day-to-day operations

### Before a demo
1. Start the Learner Lab and wait for the green dot. The EC2 instance starts with it.
2. Wait about 1–2 minutes for the container to load its models.
3. Open http://3.227.65.215.

### SSH in
```bash
ssh -i ~/.ssh/food-rec-key.pem ec2-user@3.227.65.215
```
If SSH times out, your public IP has probably changed (new Wi-Fi network). Re-allow it:
```bash
aws ec2 authorize-security-group-ingress --group-id sg-0ff69d8332a961c6a \
  --protocol tcp --port 22 --cidr $(curl -s https://checkip.amazonaws.com)/32
```

### Deploy a code change
```bash
git push                                   # on your laptop
ssh -i ~/.ssh/food-rec-key.pem ec2-user@3.227.65.215
cd Food_Recommendation_Engine && git pull
docker build -t food-recommender:latest .
docker rm -f food_recommender
# re-run the `docker run` command from section 2.6
```

### Useful commands (on the instance)
```bash
docker ps                       # is it running / healthy?
docker logs --tail 50 food_recommender
docker restart food_recommender
free -h; df -h /                # memory / disk
```

---

## 4. Cost (from the $50 Learner Lab budget)

| Item | Price (us-east-1) | Note |
|---|---|---|
| t3.large | ~$0.083 / hour | charged only while the lab session runs |
| 30 GB gp3 EBS | ~$2.40 / month | charged even while the instance is stopped |
| Elastic IP | ~$0.005 / hour (~$3.60 / month) | charged even while the instance is stopped |

Example: 40 hours of sessions over a month ≈ $3.3 + $2.4 + $3.6 ≈ **$9**.
Check *Used $X of $50* on the lab page.

---

## 5. Tear down (after grading)

```bash
export AWS_DEFAULT_REGION=us-east-1
aws ec2 terminate-instances --instance-ids i-0b181282806c5c6f6      # also deletes the 30 GB disk
aws ec2 wait instance-terminated --instance-ids i-0b181282806c5c6f6
aws ec2 release-address --allocation-id eipalloc-0fb4be8fb479bb001
aws ec2 delete-security-group --group-id sg-0ff69d8332a961c6a
aws ec2 delete-key-pair --key-name food-rec-key
```
