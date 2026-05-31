import os
from dotenv import load_dotenv

load_dotenv()

# JWT
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production-use-a-real-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Google OAuth
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./brain_tumor.db")

# Uploads
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "10"))

# Hugging Face checkpoint
HF_CHECKPOINT_REPO = os.getenv("HF_CHECKPOINT_REPO", "LuxeFats/FasterRCNN-Checkpoint")
HF_CHECKPOINT_FILENAME = os.getenv("HF_CHECKPOINT_FILENAME", "model.pth")
