# 🏥 Agentic AI Health Symptom Checker

> **Problem Statement #28** — Powered by **IBM Granite** via **IBM watsonx.ai (IBM Cloud Lite)**

An Agentic AI-driven health assistant that analyzes symptoms described in natural language, provides educational guidance sourced from WHO, CDC, and NIH, and empowers users to make informed health decisions.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **IBM Granite AI** | Natural language symptom analysis using IBM Granite via watsonx.ai |
| 🔍 **Agentic Tool-Use** | ReAct-style agent: symptom lookup, urgency assessment, language detection |
| 🌍 **Multi-language** | Supports English, Hindi, Spanish, French, Chinese, Arabic, Portuguese |
| 🚨 **Emergency Detection** | Instant red-flag detection for life-threatening symptoms |
| 📚 **Verified Sources** | Medical data from WHO, CDC, NIH, and medical journals |
| 💬 **Chat Interface** | Conversational UI with session history |
| 🛡️ **Safe by Design** | Educational only — no self-diagnosis; always recommends professionals |

---

## 🏗️ Architecture

```
SymptomsChecker/
├── app.py              # Flask API + routes
├── medical_agent.py    # IBM Granite agent (ReAct loop + tool-use)
├── symptom_db.py       # Curated symptom database (WHO/CDC/NIH)
├── templates/
│   └── index.html      # Chat UI
├── static/
│   ├── style.css       # IBM Carbon-inspired styling
│   └── app.js          # Frontend JS
├── .env                # IBM Cloud credentials (never commit!)
├── .env.example        # Credential template
├── requirements.txt    # Python dependencies
└── README.md
```

### Agentic AI Flow (ReAct Pattern)

```
User Input → [Tool: detect_language] → [Tool: check_emergency]
           → [Tool: lookup_symptoms] → [Tool: assess_urgency]
           → IBM Granite (enriched reasoning)
           → Structured Health Guidance Response
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- IBM Cloud Lite account ([Register Free](https://cloud.ibm.com/registration))
- IBM watsonx.ai project

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure IBM Cloud Credentials

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
IBM_API_KEY=your_ibm_cloud_api_key_here
IBM_PROJECT_ID=your_watsonx_project_id_here
IBM_WATSONX_URL=https://us-south.ml.cloud.ibm.com
IBM_MODEL_ID=ibm/granite-13b-instruct-v2
```

#### Getting IBM Cloud Credentials:
1. **API Key**: IBM Cloud → Manage → IAM → API Keys → Create
2. **Project ID**: watsonx.ai → Projects → (your project) → Manage → General
3. **URL**: Choose based on your region (default: US South)

### 4. Run the Application

```bash
python app.py
```

Open: **http://localhost:5000**

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `POST` | `/api/chat` | Analyze symptoms (JSON: `{"message": "..."}`) |
| `GET` | `/api/health` | Service health check |
| `GET` | `/api/symptoms` | List analyzable symptoms |
| `GET` | `/api/disclaimer` | Medical disclaimer text |
| `POST` | `/api/clear` | Clear conversation session |

### Example API Call:

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I have a sore throat and fever"}'
```

Response:
```json
{
  "response": "Based on your symptoms...",
  "urgency": "moderate",
  "urgency_info": { "label": "🟡 Moderate Urgency", "description": "..." },
  "matched_symptoms": ["sore throat", "fever"],
  "is_emergency": false,
  "language": "en",
  "source": "ibm_granite"
}
```

---

## 🌐 Supported Symptoms (Keyword Detection)

The database includes structured knowledge for:
- Fever, Sore Throat, Headache, Cough
- Chest Pain, Shortness of Breath
- Stomach Pain, Fatigue, Nausea, Rash

IBM Granite enriches responses beyond the database for any health query.

---

## 🛡️ Safety & Disclaimer

> ⚠️ **This tool is for educational purposes only.** It does NOT provide medical diagnosis, treatment, or prescriptions. Always consult a licensed healthcare professional for medical concerns. In emergencies, call **911 / 112 / 108** immediately.

---

## 🔄 Fallback Mode

If IBM Cloud credentials are not configured, the app runs in **database-only mode** using the curated WHO/CDC/NIH symptom database — no IBM account required for testing.

---

## 🧪 Technology Stack

| Layer | Technology |
|---|---|
| **AI Model** | IBM Granite 13B Instruct v2 |
| **AI Platform** | IBM watsonx.ai (IBM Cloud Lite) |
| **Backend** | Python 3.9+, Flask 3.0 |
| **Frontend** | HTML5, CSS3, Vanilla JS |
| **Medical Data** | WHO, CDC, NIH curated database |
| **Agent Pattern** | ReAct (Reason + Act) with Tool-Use |

---

*Built for EdunetIntern · Problem Statement #28 · IBM Cloud Lite*
