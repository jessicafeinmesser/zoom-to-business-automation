import os
import logging
import hmac
import hashlib
import json
import time
import tempfile
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ZOOM_ACCOUNT_ID = os.getenv("ZOOM_ACCOUNT_ID")
ZOOM_CLIENT_ID = os.getenv("ZOOM_CLIENT_ID")
ZOOM_CLIENT_SECRET = os.getenv("ZOOM_CLIENT_SECRET")
ZOOM_WEBHOOK_SECRET = os.getenv("ZOOM_WEBHOOK_SECRET")
GHL_API_KEY = os.getenv("GHL_API_KEY")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID")
GHL_BASE_URL = "https://api.gohighlevel.com/v1"
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# EXCLUSION LIST
HOST_EMAILS = ["support@fullbookai.com", "ofer.rapaport@example.com"]

app = FastAPI()

# ------------------------------------------------------------------------------
# ZOOM & GHL HELPERS
# ------------------------------------------------------------------------------

def get_zoom_access_token():
    url = f"https://zoom.us/oauth/token?grant_type=account_credentials&account_id={ZOOM_ACCOUNT_ID}"
    try:
        response = requests.post(url, auth=(ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET))
        return response.json().get("access_token")
    except: return None

def get_guest_email_from_zoom(meeting_uuid: str) -> Optional[str]:
    token = get_zoom_access_token()
    if not token: return None
    safe_uuid = meeting_uuid.replace("/", "%2F").replace("//", "%2F%2F")
    url = f"https://api.zoom.us/v2/report/meetings/{safe_uuid}/participants"
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers)
        participants = response.json().get("participants", [])
        for p in participants:
            email = p.get("user_email")
            if email and email.lower() not in [e.low
