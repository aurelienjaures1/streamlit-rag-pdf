
# 🤖 Chatbot RAG PDF avec Streamlit

Ce projet Streamlit permet de poser des questions à un chatbot basé sur vos documents PDF grâce à OpenAI et Supabase.

##  Fonctionnalités

- Upload de PDF
- Vectorisation avec OpenAI Embeddings
- Stockage dans Supabase
- Interrogation des documents via RAG (LangChain)

## 🔧 Déploiement

### Fichiers nécessaires
- `app.py`
- `requirements.txt`
- `.streamlit/secrets.toml` (non versionné)

### Secrets (via Streamlit Cloud)
```toml
OPENAI_API_KEY = "sk-..."
SUPABASE_URL = "https://xxxxx.supabase.co"
SUPABASE_SERVICE_KEY = "eyJ..."

