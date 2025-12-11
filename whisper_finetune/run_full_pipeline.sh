#!/bin/bash
# Full Pipeline: Data Generation → Training → Testing → Evaluation
# Run this after training data generation completes

set -e  # Exit on error

echo "======================================================================"
echo "WHISPER ENHANCEMENT: FULL PIPELINE"
echo "======================================================================"

cd "$(dirname "$0")"

# Check if training data exists
if [ ! -f "training_data/enhanced_labeled_arguments.json" ]; then
    echo "❌ Training data not found!"
    echo "   Wait for: python enhanced_emotion_classifier.py generate 200"
    exit 1
fi

echo "✅ Training data found"

# Step 1: Train the classifier
echo ""
echo "======================================================================"
echo "STEP 1: Training Classifier"
echo "======================================================================"
python enhanced_emotion_classifier.py train

# Check if model was created
if [ ! -f "models/enhanced_argument_classifier.pt" ]; then
    echo "❌ Training failed - model not created"
    exit 1
fi

echo "✅ Model trained successfully"

# Step 2: Generate test set (500 samples)
echo ""
echo "======================================================================"
echo "STEP 2: Generating Test Set (500 samples)"
echo "======================================================================"

if [ ! -f "test_data/test_set.json" ]; then
    python comprehensive_evaluation.py generate_test 500
else
    echo "✅ Test set already exists, skipping generation"
fi

# Step 3: Run comprehensive evaluation
echo ""
echo "======================================================================"
echo "STEP 3: Running Comprehensive Evaluation"
echo "======================================================================"
python comprehensive_evaluation.py evaluate

# Step 4: Run interactive tests
echo ""
echo "======================================================================"
echo "STEP 4: Testing on Examples"
echo "======================================================================"
python test_classifier.py examples

# Summary
echo ""
echo "======================================================================"
echo "✅ PIPELINE COMPLETE!"
echo "======================================================================"
echo ""
echo "Generated files:"
echo "  - models/enhanced_argument_classifier.pt  (Trained model)"
echo "  - test_data/test_set.json                 (500 test samples)"
echo "  - TEST_RESULTS.md                         (Comprehensive evaluation)"
echo ""
echo "Next steps:"
echo "  1. Read TEST_RESULTS.md for detailed metrics"
echo "  2. Test interactively: python test_classifier.py interactive"
echo "  3. Copy model to Pi for deployment"
echo ""
