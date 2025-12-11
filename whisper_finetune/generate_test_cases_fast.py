#!/usr/bin/env python3
"""
Fast Test Case Generation (No LLM needed!)

Generates test cases using templates - instant and deterministic
"""

import json
import random
from pathlib import Path
from enhanced_emotion_classifier import EMOTIONS, UNCERTAINTY_MARKERS, CONFIDENCE_MARKERS

# Templates for each emotion
TEMPLATES = {
    "calm": [
        "Let's {verb} the evidence on {topic} objectively and consider both perspectives carefully.",
        "We should {verb} {topic} rationally, examining the facts without bias.",
        "It's important to {verb} {topic} calmly and weigh all the information available.",
        "I think we can {verb} {topic} thoughtfully if we look at the data together.",
        "Perhaps we should {verb} {topic} systematically to reach a fair conclusion.",
    ],
    "confident": [
        "I'm absolutely certain that {topic} is clearly {adjective}. The evidence definitely supports this.",
        "There's no question that {topic} is obviously {adjective}. The data proves it without a doubt.",
        "{topic} is undeniably {adjective}. I know this for a fact based on the research.",
        "It's crystal clear that {topic} is {adjective}. Anyone can see the evidence is overwhelming.",
        "I'm 100% sure that {topic} is definitely {adjective}. The facts speak for themselves.",
    ],
    "defensive": [
        "Actually, that's not what I meant about {topic}. You're misunderstanding my point here.",
        "In my defense, I was just trying to explain that {topic} is more {adjective} than you think.",
        "You don't understand - I'm saying {topic} is {adjective}, not what you're claiming I said.",
        "That's not fair. I clearly stated that {topic} is {adjective}. You're twisting my words.",
        "Look, I already explained this. {topic} is {adjective}. Why are you still questioning me?",
    ],
    "dismissive": [
        "That's ridiculous. Anyone with basic knowledge knows that {topic} is {adjective}.",
        "Come on, really? It's obvious that {topic} is {adjective}. This isn't complicated.",
        "You can't be serious about {topic}. That's clearly {adjective}. Use some common sense.",
        "Whatever. {topic} is obviously {adjective}. I'm not going to waste time explaining this.",
        "That's absurd. {topic} is {adjective}. How can you not see that?",
    ],
    "passionate": [
        "I truly believe that {topic} is absolutely {adjective}! This is crucial for our future!",
        "We must understand that {topic} is {adjective}! It's vital that we take action now!",
        "{topic} is incredibly {adjective}! I'm deeply committed to this cause!",
        "This is so important - {topic} is fundamentally {adjective}! We can't ignore this!",
        "I'm absolutely convinced that {topic} is {adjective}! This matters more than anything!",
    ],
    "frustrated": [
        "I've already explained three times that {topic} is {adjective}. Why don't you get it?",
        "How many times do I have to say this? {topic} is {adjective}. This is pointless.",
        "Ugh, {topic} is clearly {adjective}. Are you even listening to me?",
        "This is getting ridiculous. {topic} is {adjective}. I'm done repeating myself.",
        "For the last time, {topic} is {adjective}. Why is this so hard to understand?",
    ],
    "angry": [
        "You're completely wrong! {topic} is {adjective} and that's absolutely absurd!",
        "That's nonsense! {topic} is clearly {adjective}! I'm done with this!",
        "{topic} is {adjective}! How dare you suggest otherwise! This is insulting!",
        "You have no idea what you're talking about! {topic} is {adjective}! Period!",
        "This is outrageous! {topic} is obviously {adjective}! I'm not listening to this anymore!",
    ],
    "sarcastic": [
        "Oh sure, because {topic} is definitely {adjective}. Wow, what brilliant logic there.",
        "Yeah right, {topic} is {adjective}. That makes perfect sense. Great observation.",
        "Oh absolutely, {topic} is totally {adjective}. How insightful of you.",
        "Sure thing, {topic} is {adjective}. What a groundbreaking argument you've made.",
        "Oh wow, {topic} is {adjective}? I'm amazed by your reasoning skills.",
    ],
}

TOPICS = [
    "climate change", "AI regulation", "universal basic income",
    "space exploration", "remote work", "electric vehicles",
    "cryptocurrency", "social media age limits", "nuclear energy",
    "genetic engineering", "privacy vs security", "college education",
    "immigration policy", "minimum wage", "healthcare reform",
    "gun control", "free speech online", "animal rights",
    "death penalty", "abortion policy", "tax reform",
    "drug legalization", "voting rights", "police reform",
]

VERBS = ["examine", "analyze", "discuss", "evaluate", "assess", "review", "consider", "study"]
ADJECTIVES = [
    "important", "complex", "beneficial", "problematic", "effective",
    "necessary", "controversial", "significant", "critical", "relevant"
]

def generate_test_set(num_samples=100):
    """Generate test cases using templates"""

    print(f"🚀 Generating {num_samples} test cases (FAST - no LLM needed)...")

    emotion_names = list(EMOTIONS.keys())
    samples_per_emotion = num_samples // len(emotion_names)
    extra_samples = num_samples % len(emotion_names)

    test_data = []

    for emotion_idx, emotion_name in enumerate(emotion_names):
        n_samples = samples_per_emotion + (1 if emotion_idx < extra_samples else 0)

        print(f"  {emotion_name:12}: generating {n_samples} samples...", end=" ")

        templates = TEMPLATES[emotion_name]

        for i in range(n_samples):
            # Pick random template and fill in
            template = random.choice(templates)
            topic = random.choice(TOPICS)
            verb = random.choice(VERBS)
            adjective = random.choice(ADJECTIVES)

            text = template.format(topic=topic, verb=verb, adjective=adjective)

            # Compute labels based on actual markers in text
            text_lower = text.lower()
            uncertainty_count = sum(1 for marker in UNCERTAINTY_MARKERS if marker in text_lower)
            confidence_count = sum(1 for marker in CONFIDENCE_MARKERS if marker in text_lower)

            uncertainty_score = min(uncertainty_count / 3.0, 1.0)
            confidence_score = min(confidence_count / 3.0, 1.0)

            test_data.append({
                "id": f"test_{emotion_name}_{i:03d}",
                "text": text,
                "topic": topic,
                "true_emotion": emotion_name,
                "true_emotion_idx": emotion_idx,
                "true_uncertainty": uncertainty_score,
                "true_confidence": confidence_score,
            })

        print("✓")

    # Save test set
    output_path = Path("test_data/test_set.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=2)

    print(f"\n✅ Test set saved: {output_path}")
    print(f"   Total samples: {len(test_data)}")

    # Statistics
    print(f"\n📊 Test Set Statistics:")
    for emotion_name in emotion_names:
        count = sum(1 for s in test_data if s['true_emotion'] == emotion_name)
        print(f"   {emotion_name:12}: {count} samples")

    return test_data

if __name__ == "__main__":
    import sys

    num_samples = 100
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])

    generate_test_set(num_samples)
    print("\n✅ Done! Now run:")
    print("   python comprehensive_evaluation.py evaluate")
