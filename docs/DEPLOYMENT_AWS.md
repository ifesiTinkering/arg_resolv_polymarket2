# AWS EC2 Deployment Guide

Complete guide for deploying the Argument Resolver server on AWS EC2.

## Instance Requirements

| Specification | Minimum | Recommended |
|--------------|---------|-------------|
| Instance Type | t2.medium | t2.large |
| vCPUs | 2 | 2+ |
| RAM | 4 GB | 8 GB |
| Storage | 20 GB EBS | 50 GB EBS |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

## AWS Setup

### 1. Launch EC2 Instance

1. Go to AWS Console > EC2 > Launch Instance
2. Configure:
   - **Name**: `argument-resolver-server`
   - **AMI**: Ubuntu Server 22.04 LTS (64-bit x86)
   - **Instance type**: t2.large
   - **Key pair**: Create new or use existing
   - **Network settings**: Allow SSH (22), HTTP (80), Custom TCP (7863, 7864)

### 2. Configure Security Group

Create inbound rules:

| Type | Port Range | Source | Description |
|------|------------|--------|-------------|
| SSH | 22 | Your IP | SSH access |
| Custom TCP | 7863 | 0.0.0.0/0 | Gradio Web UI |
| Custom TCP | 7864 | 0.0.0.0/0 | Results Receiver API |
| HTTP | 80 | 0.0.0.0/0 | Optional: nginx proxy |

### 3. Allocate Elastic IP (Optional but Recommended)

1. EC2 > Elastic IPs > Allocate Elastic IP address
2. Associate with your instance
3. This gives you a static IP that doesn't change on restart

## Server Setup

### 1. Connect to Instance

```bash
# Using your key pair
ssh -i "your-key.pem" ubuntu@<ec2-public-ip>
```

### 2. Update System

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv ffmpeg git htop
```

### 3. Clone Repository

```bash
cd ~
git clone https://github.com/ifesiTinkering/arg_resolv_polymarket.git
cd arg_resolv_polymarket
```

### 4. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 5. Install Dependencies

```bash
# Install PyTorch (CPU version for cost efficiency)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install project dependencies
pip install -r requirements.txt
```

### 6. Configure Environment

```bash
nano .env
```

Add:
```bash
HUGGINGFACE_TOKEN=hf_your_token_here

# Optional
POE_API_KEY=your_poe_key_here
```

### 7. Copy Model File

Transfer the trained emotion model:
```bash
# From your local machine
scp -i "your-key.pem" models/enhanced_argument_classifier.pt ubuntu@<ec2-ip>:~/arg_resolv_polymarket/models/
```

Or download if hosted:
```bash
mkdir -p models
# wget or curl your model file
```

## Running the Services

### Option A: Manual Start (Development)

```bash
cd ~/arg_resolv_polymarket
source venv/bin/activate

# Terminal 1: Start Results Receiver
python results_receiver.py

# Terminal 2: Start Web UI
python browse_arguments.py
```

### Option B: Systemd Services (Production)

Create service files:

**Results Receiver Service:**
```bash
sudo nano /etc/systemd/system/argument-receiver.service
```

```ini
[Unit]
Description=Argument Resolver Results Receiver
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/arg_resolv_polymarket
Environment=PATH=/home/ubuntu/arg_resolv_polymarket/venv/bin
ExecStart=/home/ubuntu/arg_resolv_polymarket/venv/bin/python results_receiver.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Web UI Service:**
```bash
sudo nano /etc/systemd/system/argument-ui.service
```

```ini
[Unit]
Description=Argument Resolver Web UI
After=network.target argument-receiver.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/arg_resolv_polymarket
Environment=PATH=/home/ubuntu/arg_resolv_polymarket/venv/bin
ExecStart=/home/ubuntu/arg_resolv_polymarket/venv/bin/python browse_arguments.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable argument-receiver argument-ui
sudo systemctl start argument-receiver argument-ui

# Check status
sudo systemctl status argument-receiver
sudo systemctl status argument-ui
```

### Option C: Using Screen (Quick Testing)

```bash
# Start results receiver
screen -S receiver
cd ~/arg_resolv_polymarket && source venv/bin/activate
python results_receiver.py
# Press Ctrl+A, D to detach

# Start web UI
screen -S webui
cd ~/arg_resolv_polymarket && source venv/bin/activate
python browse_arguments.py
# Press Ctrl+A, D to detach

# Reattach later
screen -r receiver
screen -r webui
```

## Nginx Reverse Proxy (Optional)

For cleaner URLs and SSL support:

```bash
sudo apt install nginx
sudo nano /etc/nginx/sites-available/argument-resolver
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    # Web UI
    location / {
        proxy_pass http://127.0.0.1:7863;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Results API
    location /api/ {
        proxy_pass http://127.0.0.1:7864/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Enable:
```bash
sudo ln -s /etc/nginx/sites-available/argument-resolver /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## SSL with Let's Encrypt (Optional)

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

## Accessing the System

- **Web UI**: `http://<ec2-public-ip>:7863`
- **API Health Check**: `http://<ec2-public-ip>:7864/health`

Update Pi's `.env` with the server IP:
```bash
LAPTOP_IP=<ec2-public-ip>
```

## Monitoring

### View Logs

```bash
# Systemd logs
journalctl -u argument-receiver -f
journalctl -u argument-ui -f

# Combined
journalctl -u argument-receiver -u argument-ui -f
```

### Resource Usage

```bash
# CPU and Memory
htop

# Disk usage
df -h

# Network connections
netstat -tlnp
```

### Health Checks

```bash
# Check if services are running
curl http://localhost:7864/health
curl http://localhost:7863
```

## Database Management

### Backup Arguments Database

```bash
# Create backup
tar -czvf arguments_backup_$(date +%Y%m%d).tar.gz arguments_db/

# Copy to S3 (optional)
aws s3 cp arguments_backup_*.tar.gz s3://your-bucket/backups/
```

### Restore from Backup

```bash
tar -xzvf arguments_backup_20251201.tar.gz
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
journalctl -u argument-receiver -n 50 --no-pager

# Check Python environment
/home/ubuntu/arg_resolv_polymarket/venv/bin/python --version
```

### Port Already in Use

```bash
# Find process using port
sudo lsof -i :7863
sudo lsof -i :7864

# Kill if needed
sudo kill -9 <PID>
```

### Memory Issues

```bash
# Check memory
free -h

# Add swap if needed
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make permanent
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Connection Refused from Pi

1. Check security group allows port 7864
2. Check service is running: `systemctl status argument-receiver`
3. Check firewall: `sudo ufw status`

## Cost Optimization

### Use Spot Instances

For development/testing, spot instances can reduce costs by 60-90%:
- Request spot instance with same specs
- Set maximum price slightly above current spot price
- Note: Instance may be terminated with 2-minute warning

### Auto-Shutdown

Schedule shutdown during off-hours:
```bash
# Shutdown at 11 PM, start at 7 AM
sudo crontab -e
0 23 * * * /sbin/shutdown -h now

# Use Lambda + CloudWatch to auto-start
```

### Right-Size Instance

Monitor usage and downgrade if:
- CPU consistently < 20%
- Memory consistently < 50%

## Updating the System

```bash
cd ~/arg_resolv_polymarket
source venv/bin/activate

# Pull updates
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart services
sudo systemctl restart argument-receiver argument-ui
```

## File Locations

| Path | Description |
|------|-------------|
| `/home/ubuntu/arg_resolv_polymarket/` | Main project |
| `/home/ubuntu/arg_resolv_polymarket/venv/` | Virtual environment |
| `/home/ubuntu/arg_resolv_polymarket/arguments_db/` | Stored arguments |
| `/home/ubuntu/arg_resolv_polymarket/models/` | ML models |
| `/etc/systemd/system/argument-*.service` | Service files |
| `/etc/nginx/sites-available/argument-resolver` | Nginx config |
