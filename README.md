# ATHENA VAR

**ATHENA VAR** is an AI-powered Video Assistant Referee (VAR) system for football/soccer. It uses computer vision and deep learning to automatically detect offsides and fouls in match footage, providing accurate, real-time decisions without human bias.

---

## Features

- **Offside Detection** — Automatically identifies offside positions using player tracking and line projection
- **Foul Detection** — Classifies illegal contact and dangerous play using a trained classifier
- **Team Detection** — Distinguishes between teams and individual players across frames
- **Decision Engine** — Aggregates CV pipeline outputs into a final VAR ruling
- **REST API** — FastAPI backend for integrating with external systems or frontends
- **React Frontend** — Clean UI for visualizing detections and decisions in real time

---

## Project Structure

```
athena-var/
├── src/
│   └── athena/
│       ├── api/          # FastAPI backend
│       ├── cv_pipeline/  # Computer vision: detection, offside, team classifier
│       ├── classifier/   # Foul classification models
│       ├── decision/     # Decision engine logic
│       └── ingestion/    # Video input and frame handling
├── frontend/             # React + Vite frontend
├── tests/                # Test suite
├── main.py               # Entry point
├── pyproject.toml        # Python project config
└── uv.lock               # Locked dependencies (uv)
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

---

### 1. Clone the Repository

```bash
git clone https://github.com/ConnectRaheem/athena-var.git
cd athena-var
```

---

### 2. Backend Setup

**Using uv (recommended):**
```bash
uv sync
```

**Using pip:**
```bash
pip install -e .
```

**Environment variables:**

Create a `.env` file in the root directory:
```env
# Add your environment variables here
# Example:
# API_KEY=your_key_here
# MODEL_PATH=path/to/weights
```

> See `.env.example` for all required variables. *(coming soon)*

---

### 3. Model Weights

ATHENA VAR uses YOLO-based models for player and ball detection.

> **Note:** Model weights (`*.pt` files) are not included in this repository.
> Download instructions and links will be added here soon.

Place downloaded weights in the project root or update the relevant path in your `.env`.

---

### 4. Run the Backend

```bash
python main.py
```

Or start the API server directly:
```bash
uvicorn src.athena.api.main:app --reload
```

The API will be available at `http://localhost:8000`.

---

### 5. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`.

---

## API Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Submit a video clip for VAR analysis |
| `GET`  | `/decision/{id}` | Retrieve a VAR decision by ID |
| `GET`  | `/health` | Health check |

> Full API docs available at `http://localhost:8000/docs` once the server is running.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| CV / ML | YOLO, OpenCV |
| Backend | Python, FastAPI |
| Frontend | React, Vite, Tailwind CSS |
| Package Manager | uv (Python), npm (Node) |

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## License

This project is currently unlicensed. License to be added.

---

> Built by [ConnectRaheem](https://github.com/ConnectRaheem)
