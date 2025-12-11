# System Architecture

Comprehensive documentation of the Argument Resolver system architecture, components, and data flow.

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Emotion Classifier](#emotion-classifier)
5. [RAG Knowledge Base](#rag-knowledge-base)
6. [Polymarket Integration](#polymarket-integration)
7. [Fact-Checking Pipeline](#fact-checking-pipeline)
8. [Storage System](#storage-system)
9. [Web Interface](#web-interface)

---

## System Overview

The Argument Resolver is a distributed edge-cloud system for real-time conversation analysis. It combines speech processing, emotion detection, and multi-source fact-checking.

### Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TIER 1: EDGE                                   │
│                           (Raspberry Pi 4)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Audio Capture → Diarization → Transcription → HTTP POST            │   │
│  │      (USB)       (pyannote)      (Whisper)      (to cloud)          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Processing Time: 4-6 seconds                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TIER 2: CLOUD                                  │
│                            (AWS EC2 t2.large)                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  FastAPI Server → Emotion Analysis → Fact-Checking → Storage        │   │
│  │    (receive)      (classifier)        (3 sources)    (JSON)         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│  Processing Time: 2-3 seconds                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TIER 3: CLIENT                                 │
│                              (Web Browser)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Gradio Web UI with chat bubbles, emotion badges, fact panels       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Edge Processing**: Heavy ML tasks (diarization, transcription) on Pi to reduce cloud costs
2. **Offline-First**: Pi stores locally even when server is unavailable
3. **Multi-Source Verification**: Never rely on a single fact source
4. **Low Latency**: End-to-end processing in 6-10 seconds

---

## Component Architecture

### File Structure

```
arg_resolv_polymarket/
│
├── EDGE COMPONENTS (Raspberry Pi)
│   ├── pi_record_and_process.py     # Main recording script
│   └── argument_processing.py        # Shared processing functions
│
├── CLOUD COMPONENTS (AWS EC2)
│   ├── results_receiver.py           # FastAPI server
│   ├── emotion_classifier.py         # 8-class emotion model
│   ├── segment_fact_checker.py       # Fact-check orchestrator
│   ├── knowledge_base.py             # RAG semantic search
│   ├── polymarket_client.py          # Prediction market API
│   └── storage.py                    # Database manager
│
├── CLIENT COMPONENTS
│   └── browse_arguments.py           # Gradio web interface
│
├── DATA
│   ├── knowledge_base.json           # Curated facts (25+)
│   ├── models/
│   │   └── enhanced_argument_classifier.pt
│   └── arguments_db/                 # Stored arguments
│
└── CONFIG
    ├── requirements.txt
    └── .env
```

### Component Dependencies

```
pi_record_and_process.py
    └── argument_processing.py
            ├── pyannote.audio (diarization)
            └── whisper (transcription)

results_receiver.py
    ├── emotion_classifier.py
    │       └── sentence-transformers
    ├── segment_fact_checker.py
    │       ├── knowledge_base.py
    │       │       └── sentence-transformers
    │       ├── polymarket_client.py
    │       │       └── requests
    │       └── web_search (DuckDuckGo)
    │               └── beautifulsoup4
    └── storage.py

browse_arguments.py
    └── gradio
```

---

## Data Flow

### Complete Processing Pipeline

```
Step 1: AUDIO CAPTURE (Pi)
┌─────────────────────────────────────────┐
│  USB Microphone                         │
│      ↓                                  │
│  Record 30 seconds                      │
│      ↓                                  │
│  Save as 16kHz mono WAV                 │
└─────────────────────────────────────────┘
            ↓
Step 2: SPEAKER DIARIZATION (Pi)
┌─────────────────────────────────────────┐
│  pyannote/speaker-diarization-3.1       │
│      ↓                                  │
│  Identify speakers & timestamps         │
│      ↓                                  │
│  Output: [(speaker, start, end), ...]   │
└─────────────────────────────────────────┘
            ↓
Step 3: TRANSCRIPTION (Pi)
┌─────────────────────────────────────────┐
│  For each segment:                      │
│      Extract audio slice                │
│      ↓                                  │
│      Whisper STT                        │
│      ↓                                  │
│      Add to transcript                  │
└─────────────────────────────────────────┘
            ↓
Step 4: HTTP TRANSFER (Pi → AWS)
┌─────────────────────────────────────────┐
│  POST /receive_results                  │
│      - audio.wav                        │
│      - transcript.txt                   │
│      - metadata.json                    │
└─────────────────────────────────────────┘
            ↓
Step 5: EMOTION ANALYSIS (AWS)
┌─────────────────────────────────────────┐
│  For each segment:                      │
│      SentenceTransformer encode         │
│      ↓                                  │
│      Linear classifier                  │
│      ↓                                  │
│      Emotion + confidence               │
└─────────────────────────────────────────┘
            ↓
Step 6: FACT-CHECKING (AWS)
┌─────────────────────────────────────────┐
│  For each segment (parallel):           │
│      ├── RAG Knowledge Base             │
│      ├── Polymarket API                 │
│      └── Web Search                     │
│              ↓                          │
│      Aggregate results                  │
└─────────────────────────────────────────┘
            ↓
Step 7: STORAGE (AWS)
┌─────────────────────────────────────────┐
│  Save to arguments_db/                  │
│      - audio.wav                        │
│      - metadata.json (with analysis)    │
│      - transcript.txt                   │
└─────────────────────────────────────────┘
            ↓
Step 8: DISPLAY (Browser)
┌─────────────────────────────────────────┐
│  Gradio Web UI                          │
│      - Chat bubbles                     │
│      - Emotion badges                   │
│      - Fact-check hover panels          │
└─────────────────────────────────────────┘
```

---

## Emotion Classifier

### Architecture

```
┌────────────────────────────────────────────────────────────┐
│                   EMOTION CLASSIFIER                        │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: "You're completely wrong about this!"               │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  SentenceTransformer (all-MiniLM-L6-v2)             │   │
│  │  - 66 million parameters                             │   │
│  │  - FROZEN (not trained)                              │   │
│  │  - Output: 384-dimensional vector                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                         ↓                                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Shared Layer                                        │   │
│  │  - Linear(384 → 128)                                 │   │
│  │  - ReLU activation                                   │   │
│  │  - Dropout(0.3)                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│           ↓               ↓               ↓                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Emotion     │  │ Uncertainty │  │ Confidence  │        │
│  │ Head        │  │ Head        │  │ Head        │        │
│  │ 128→64→8    │  │ 128→64→1    │  │ 128→64→1    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│        ↓               ↓               ↓                   │
│    Softmax         Sigmoid         Sigmoid                 │
│        ↓               ↓               ↓                   │
│   8 emotions        0-1            0-1                     │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

### 8 Emotion Classes

| # | Emotion | Description | Linguistic Markers |
|---|---------|-------------|-------------------|
| 0 | calm | Measured, rational | "let's consider", "objectively" |
| 1 | confident | Assertive, certain | "definitely", "clearly" |
| 2 | defensive | Protective, justifying | "that's not what I meant" |
| 3 | dismissive | Condescending | "that's ridiculous" |
| 4 | passionate | Enthusiastic | "this is crucial" |
| 5 | frustrated | Annoyed, impatient | "I've already explained" |
| 6 | angry | Heated, hostile | "you're wrong" |
| 7 | sarcastic | Mocking, ironic | "oh sure", "yeah right" |

### Training Details

- **Data**: 500 synthetic samples from GPT-4 (balanced across 8 classes)
- **Split**: 400 train / 100 validation
- **Optimizer**: Adam, lr=0.001
- **Epochs**: 20
- **Loss**: 0.3*MSE(uncertainty) + 0.3*MSE(confidence) + 0.4*CrossEntropy(emotion)
- **Result**: 73.2% accuracy

---

## RAG Knowledge Base

### How It Works

```
OFFLINE INDEXING (at startup)
┌─────────────────────────────────────────────────────────┐
│  knowledge_base.json (25+ facts)                        │
│       ↓                                                 │
│  SentenceTransformer.encode()                           │
│       ↓                                                 │
│  fact_embeddings: Tensor[25, 384]                       │
└─────────────────────────────────────────────────────────┘

ONLINE SEARCH (per query)
┌─────────────────────────────────────────────────────────┐
│  Query: "Remote work increases productivity"            │
│       ↓                                                 │
│  SentenceTransformer.encode() → query_embedding[384]    │
│       ↓                                                 │
│  cosine_similarity(query_embedding, fact_embeddings)    │
│       ↓                                                 │
│  Filter: similarity >= 0.3                              │
│       ↓                                                 │
│  Return top-k facts with sources                        │
└─────────────────────────────────────────────────────────┘
```

### Knowledge Base Structure

```json
{
  "facts": [
    {
      "text": "Stanford 2020: remote workers 13% more productive",
      "source": "Stanford WFH Study",
      "url": "https://...",
      "stance": "supporting",
      "category": "remote_work"
    }
  ],
  "polymarket_markets": [
    {
      "text": "Will AI replace 50% of jobs by 2030?",
      "url": "https://polymarket.com/...",
      "keywords": ["ai", "jobs", "automation"],
      "category": "ai"
    }
  ]
}
```

### Categories Covered

- Remote Work
- Climate Change
- Electric Vehicles
- AI Capabilities
- Minimum Wage
- Nuclear Energy
- Renewable Energy
- Transportation
- Healthcare
- Politics

---

## Polymarket Integration

### LRU Caching Strategy

```
┌─────────────────────────────────────────────────────────┐
│                 POLYMARKET CACHE                         │
├─────────────────────────────────────────────────────────┤
│  Preloaded: 22 markets                                  │
│  Max Size: 30 markets                                   │
│  Eviction: Least Recently Used (LRU)                    │
└─────────────────────────────────────────────────────────┘

Query Flow:
┌─────────┐    ┌──────────┐    ┌─────────────┐
│  Query  │───>│  Cache   │───>│   Return    │
│         │    │  Lookup  │ Y  │   Cached    │
└─────────┘    └────┬─────┘    └─────────────┘
                    │ N
                    ▼
           ┌──────────────┐
           │  Polymarket  │
           │     API      │
           └──────┬───────┘
                  ▼
           ┌──────────────┐
           │  Add to      │
           │  Cache       │
           └──────┬───────┘
                  ▼
           ┌──────────────┐
           │  LRU Evict   │
           │  if full     │
           └──────────────┘
```

### API Integration

```python
# Gamma API endpoint
BASE_URL = "https://gamma-api.polymarket.com"

# Search markets
GET /markets?tag=<topic>&active=true&limit=5

# Response includes:
# - question: "Will X happen by Y?"
# - outcomes: ["Yes", "No"]
# - outcomePrices: [0.63, 0.37]  # 63% Yes, 37% No
```

---

## Fact-Checking Pipeline

### Parallel Multi-Source Query

```
segment_fact_checker.py
           │
           ├──────────────────────────────────────────┐
           │                                          │
           ▼                                          │
┌─────────────────────┐                              │
│   asyncio.gather()  │                              │
├─────────────────────┤                              │
│         │           │                              │
│    ┌────┼────┐      │                              │
│    ▼    ▼    ▼      │                              │
│   KB  Poly  Web     │  ← Parallel execution        │
│    │    │    │      │                              │
│    └────┼────┘      │                              │
│         ▼           │                              │
│    Aggregate        │                              │
└─────────────────────┘                              │
           │                                          │
           ▼                                          │
┌─────────────────────────────────────────────────────┤
│  Result:                                            │
│  {                                                  │
│    "supporting": [                                  │
│      {"text": "...", "source": "KB", "url": "..."},│
│      {"text": "...", "source": "Polymarket"},      │
│      {"text": "...", "source": "DuckDuckGo"}       │
│    ],                                               │
│    "contradicting": [...]                           │
│  }                                                  │
└─────────────────────────────────────────────────────┘
```

### Web Search Implementation

```python
# DuckDuckGo HTML API (no API key required)
POST https://html.duckduckgo.com/html/
    q=<query>

# Parse HTML response with BeautifulSoup
# Extract: title, snippet, URL
# Return top 3 results
```

---

## Storage System

### Database Structure

```
arguments_db/
├── arguments.json              # Index file
└── arguments/
    ├── 20251205_145653/       # Argument folder (timestamp ID)
    │   ├── audio.wav          # Original recording
    │   ├── metadata.json      # Full analysis results
    │   └── transcript.txt     # Timestamped transcript
    └── 20251204_132845/
        ├── audio.wav
        ├── metadata.json
        └── transcript.txt
```

### Metadata Schema

```json
{
  "id": "20251205_145653",
  "timestamp": "2025-12-05T14:56:53Z",
  "duration_seconds": 30,
  "num_speakers": 2,
  "segments": [
    {
      "segment_id": 0,
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 5.2,
      "text": "I think remote work is more productive",
      "emotion": {
        "emotion": "confident",
        "emotion_confidence": 0.82,
        "uncertainty": 0.15,
        "confidence": 0.78
      },
      "facts": {
        "supporting": [
          {
            "text": "Stanford study: 13% productivity increase",
            "source_type": "knowledge_base",
            "source_name": "Stanford WFH Study",
            "url": "https://...",
            "similarity": 0.67
          }
        ],
        "contradicting": []
      }
    }
  ]
}
```

---

## Web Interface

### Gradio Components

```
browse_arguments.py
        │
        ├── Dropdown: Select argument by ID
        │
        ├── HTML: Chat bubble display
        │       ├── Speaker color coding
        │       ├── Emotion badges
        │       └── Hover fact panels
        │
        └── Audio: Playback original recording
```

### Chat Bubble Design

```html
<div class="chat-bubble speaker-0">
  <div class="speaker-label">SPEAKER_00</div>
  <div class="message">
    I think remote work is more productive
  </div>
  <div class="metadata">
    <span class="timestamp">0.0s</span>
    <span class="emotion-badge confident">CONFIDENT</span>
  </div>
  <div class="fact-panel" style="display:none">
    <!-- Shown on hover -->
    <h4>Emotion Analysis</h4>
    <p>Emotion: CONFIDENT (82%)</p>

    <h4>Supporting Facts</h4>
    <ul>
      <li>Stanford study: 13% productivity increase</li>
    </ul>
  </div>
</div>
```

---

## Performance Characteristics

### Latency Breakdown

| Component | Time | Location |
|-----------|------|----------|
| Audio Recording | 30s | Pi |
| Speaker Diarization | 2-3s | Pi |
| Whisper Transcription | 2-3s | Pi |
| Network Transfer | <1s | Pi → AWS |
| Emotion Classification | ~100ms | AWS |
| RAG Search | ~50ms | AWS |
| Polymarket Lookup | ~80ms | AWS |
| Web Search | 1-2s | AWS |
| Storage | <100ms | AWS |
| **Total** | **6-10s** | |

### Resource Usage

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| Pi (recording) | 80% | 2GB | 50MB/arg |
| AWS (processing) | 30% | 4GB | 100MB/arg |
| Browser | 10% | 200MB | - |

---

## Security Considerations

1. **API Keys**: Store in `.env`, never commit
2. **Network**: Use HTTPS in production
3. **Authentication**: Add Gradio auth for public deployment
4. **Data**: Audio files may contain sensitive conversations
5. **Rate Limiting**: Polymarket and web search have implicit limits
