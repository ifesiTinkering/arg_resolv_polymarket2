# Raspberry Pi Deployment Guide

Complete guide for setting up the Argument Resolver edge device on a Raspberry Pi.

## Hardware Requirements

| Component | Specification | Notes |
|-----------|--------------|-------|
| Raspberry Pi | Model 4B, 4GB+ RAM | 8GB recommended for faster processing |
| Storage | 32GB+ microSD | Class 10 or faster |
| Microphone | USB microphone | Tested with Blue Snowball, Samson Go Mic |
| Power | 5V 3A USB-C | Official Pi power supply recommended |
| Network | WiFi or Ethernet | For sending results to server |

## Initial Setup

### 1. Flash Raspberry Pi OS

```bash
# Download Raspberry Pi Imager from https://www.raspberrypi.com/software/
# Flash "Raspberry Pi OS (64-bit)" to your microSD card
# Enable SSH and configure WiFi in the imager settings
```

### 2. First Boot Configuration

```bash
# SSH into your Pi
ssh pi@raspberrypi.local

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y \
    python3-pip \
    python3-venv \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    git
```

### 3. Clone the Repository

```bash
cd ~
git clone https://github.com/ifesiTinkering/arg_resolv_polymarket.git
cd arg_resolv_polymarket
```

### 4. Create Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 5. Install Python Dependencies

```bash
# Install PyTorch for ARM (Pi-optimized)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# Install pyannote.audio (speaker diarization)
pip install pyannote.audio
```

### 6. Configure Environment Variables

```bash
# Create .env file
nano .env
```

Add the following:
```bash
# Required
HUGGINGFACE_TOKEN=hf_your_token_here
LAPTOP_IP=your_server_ip_here

# Optional
POE_API_KEY=your_poe_key_here
```

Get your HuggingFace token:
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with read access
3. Accept the pyannote model terms at https://huggingface.co/pyannote/speaker-diarization-3.1

### 7. Test Microphone

```bash
# List audio devices
arecord -l

# Test recording (5 seconds)
arecord -D plughw:1,0 -f cd -t wav -d 5 test.wav

# Play back
aplay test.wav

# Clean up
rm test.wav
```

If your microphone is not detected:
```bash
# Check USB devices
lsusb

# Check audio devices
cat /proc/asound/cards
```

## Running the System

### Start Recording

```bash
cd ~/arg_resolv_polymarket
source venv/bin/activate

# Run the recorder
python3 pi_record_and_process.py
```

### What Happens

1. **Audio Capture**: Records 30 seconds from USB microphone
2. **Speaker Diarization**: Identifies different speakers using pyannote.audio
3. **Transcription**: Converts speech to text using Whisper
4. **Local Storage**: Saves to `arguments_db/` on the Pi
5. **Remote Sync**: Sends results to AWS server (if available)

### Run as Background Service

Create a systemd service for automatic startup:

```bash
sudo nano /etc/systemd/system/argument-resolver.service
```

```ini
[Unit]
Description=Argument Resolver Recording Service
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/arg_resolv_polymarket
Environment=PATH=/home/pi/arg_resolv_polymarket/venv/bin
ExecStart=/home/pi/arg_resolv_polymarket/venv/bin/python3 pi_record_and_process.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable argument-resolver
sudo systemctl start argument-resolver

# Check status
sudo systemctl status argument-resolver

# View logs
journalctl -u argument-resolver -f
```

## Configuration Options

### Adjust Recording Duration

Edit `pi_record_and_process.py`:
```python
RECORD_DURATION = 30  # Change to desired seconds (e.g., 60 for 1 minute)
```

### Change Audio Device

If your microphone isn't the default device:
```python
# In pi_record_and_process.py, find the recording function and specify device
# Example: device=1 for second audio device
```

### Offline Mode

The Pi saves all recordings locally even when the server is unreachable:
- Results stored in `arguments_db/`
- Automatic retry when connection is restored

## Troubleshooting

### "No module named 'pyannote'"

```bash
source venv/bin/activate
pip install pyannote.audio --upgrade
```

### "CUDA not available" Warning

This is normal on Pi - the system uses CPU processing. Ignore this warning.

### Microphone Not Found

```bash
# Check if microphone is connected
lsusb | grep -i audio

# List ALSA devices
arecord -l

# Test with specific device
arecord -D plughw:1,0 -f cd -t wav -d 5 test.wav
```

### Out of Memory Errors

```bash
# Increase swap space
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=100 to CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Network Connection Issues

```bash
# Check connection to server
ping $LAPTOP_IP

# Test HTTP connection
curl http://$LAPTOP_IP:7864/health
```

### pyannote Token Issues

```bash
# Verify token is set
echo $HUGGINGFACE_TOKEN

# Re-authenticate
huggingface-cli login
```

## Performance Optimization

### Reduce Processing Time

1. Use smaller Whisper model:
```python
# In argument_processing.py
model = whisper.load_model("tiny")  # Instead of "base" or "small"
```

2. Reduce audio quality:
```python
SAMPLE_RATE = 16000  # Standard for speech recognition
```

### Memory Management

```bash
# Monitor memory usage
htop

# Clear Python cache periodically
find . -type d -name __pycache__ -exec rm -rf {} +
```

## File Locations

| Path | Description |
|------|-------------|
| `~/arg_resolv_polymarket/` | Main project directory |
| `~/arg_resolv_polymarket/venv/` | Python virtual environment |
| `~/arg_resolv_polymarket/arguments_db/` | Stored arguments |
| `~/arg_resolv_polymarket/.env` | Environment variables |
| `/etc/systemd/system/argument-resolver.service` | Systemd service file |

## Updating the System

```bash
cd ~/arg_resolv_polymarket
source venv/bin/activate

# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart service if running
sudo systemctl restart argument-resolver
```

## Security Notes

- Never commit `.env` file to git
- Use SSH keys instead of passwords
- Keep system packages updated
- Consider firewall rules if Pi is network-accessible
