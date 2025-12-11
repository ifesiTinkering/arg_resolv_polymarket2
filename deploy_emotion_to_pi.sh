#!/bin/bash
# Deploy Emotion Classifier to Raspberry Pi

PI_USER="ifesiras"
PI_HOST="raspberrypi.local"
PI_PASSWORD="play"
PROJECT_DIR="~/aiot_project"

echo "========================================================================"
echo "DEPLOYING EMOTION CLASSIFIER TO RASPBERRY PI"
echo "========================================================================"

# Files to copy
FILES=(
    "emotion_classifier.py"
    "test_emotion_on_pi.py"
    "models/enhanced_argument_classifier.pt"
)

echo ""
echo "Step 1: Creating models directory on Pi..."
sshpass -p "$PI_PASSWORD" ssh -o StrictHostKeyChecking=no $PI_USER@$PI_HOST "mkdir -p $PROJECT_DIR/models"

echo ""
echo "Step 2: Copying files to Pi..."
for file in "${FILES[@]}"; do
    echo "  Copying $file..."
    sshpass -p "$PI_PASSWORD" scp -o StrictHostKeyChecking=no "$file" $PI_USER@$PI_HOST:$PROJECT_DIR/$file
done

echo ""
echo "Step 3: Installing dependencies on Pi..."
sshpass -p "$PI_PASSWORD" ssh -o StrictHostKeyChecking=no $PI_USER@$PI_HOST << 'EOF'
    cd ~/aiot_project
    echo "  Installing sentence-transformers..."
    pip install --break-system-packages sentence-transformers scikit-learn
EOF

echo ""
echo "Step 4: Testing on Pi..."
sshpass -p "$PI_PASSWORD" ssh -o StrictHostKeyChecking=no $PI_USER@$PI_HOST << 'EOF'
    cd ~/aiot_project
    echo "  Running test script..."
    python3 test_emotion_on_pi.py
EOF

echo ""
echo "========================================================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "========================================================================"
echo ""
echo "The emotion classifier is now installed on your Raspberry Pi."
echo ""
echo "Next steps:"
echo "  1. Verify tests passed above"
echo "  2. Integrate into audio_processor.py (automated in next step)"
echo "  3. Test with real argument recordings"
echo ""
