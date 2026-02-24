import os
import logging
import hmac
import hashlib
import json
import time
import tempfile
import requests
import urllib.parse
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set
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
GHL_CATCH_ALL_ID = os.getenv("GHL_CATCH_ALL_CONTACT_ID")

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

HOST_EMAILS = ["support@fullbookai.com", "info@fullbookai.com", "ofer.rapaport@gmail.com"]
PROCESSED_UUIDS: Set[str] = set()

app = FastAPI()

# ------------------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------------------

def get_zoom_access_token():
    url = f"https://zoom.us/oauth/token?grant_type=account_credentials&account_id={ZOOM_ACCOUNT_ID}"
    try:
        res = requests.post(url, auth=(ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET))
        return res.json().get("access_token")
    except: return None

def get_zoom_participants(meeting_uuid: str) -> List[Dict]:
    token = get_zoom_access_token()
    if not token: return []
    encoded_uuid = urllib.parse.quote(urllib.parse.quote(meeting_uuid, safe=''), safe='')
    url = f"https://api.zoom.us/v2/report/meetings/{encoded_uuid}/participants"
    try:
        res = requests.get(url, headers={"Authorization": f"Bearer {token}"})
        return res.json().get("participants", [])
    except: return []

def find_contact_in_ghl(query: str) -> Optional[str]:
    if not query or len(query) < 2: return None
    try:
        url = f"{GHL_BASE_URL}/contacts/"
        params = {"locationId": GHL_LOCATION_ID, "query": query}
        resp = requests.get(url, headers={"Authorization": f"Bearer {GHL_API_KEY}"}, params=params)
        contacts = resp.json().get("contacts", [])
        return contacts[0]["id"] if contacts else None
    except: return None

def upload_ghl_note(contact_id: str, note_body: str):
    """Uploads note and logs the result from GHL."""
    url = f"{GHL_BASE_URL}/contacts/{contact_id}/notes"
    try:
        resp = requests.post(url, headers={"Authorization": f"Bearer {GHL_API_KEY}"}, json={"body": note_body})
        if resp.status_code in [200, 201]:
            logger.info(f"Successfully posted note to GHL ID: {contact_id}")
        else:
            logger.error(f"GHL REJECTED NOTE (Status {resp.status_code}): {resp.text}")
    except Exception as e:
        logger.error(f"Network error uploading to GHL: {e}")

# ------------------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------------------

def process_recording_logic(download_url: str, zoom_id: str, zoom_uuid: str, download_token: str):
    temp_file_path = None
    file_upload = None

    try:
        logger.info(f"--- STARTING ANALYSIS FOR MEETING {zoom_id} ---")
        
        # 1. Get Zoom Attendee Names to pass to AI
        participants = get_zoom_participants(zoom_uuid)
        attendee_names = [p.get('name') for p in participants if p.get('user_email') not in HOST_EMAILS]
        logger.info(f"Zoom Attendee Names: {attendee_names}")

        # 2. Identify Contact ID via Email (Fastest)
        contact_id = None
        for p in participants:
            email = p.get('user_email')
            if email and email.lower() not in HOST_EMAILS:
                contact_id = find_contact_in_ghl(email)
                if contact_id: break

        # 3. Gemini Processing
        # (Download logic omitted for brevity, same as previous)
        auth_url = f"{download_url}?access_token={download_token}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp_file_path = tmp.name
            with requests.get(auth_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=16384): tmp.write(chunk)
        
        file_upload = genai.upload_file(temp_file_path, mime_type="video/mp4")
        while file_upload.state.name == "PROCESSING":
            time.sleep(5)
            file_upload = genai.get_file(file_upload.name)
        
        model = genai.GenerativeModel(model_name="models/gemini-flash-latest")
        
        # IMPROVED PROMPT: We tell Gemini exactly who was in the room to avoid spelling errors
        prompt = (
            f"Analyze this recording. Here is the list of people Zoom detected: {attendee_names}. "
            f"Identify the primary client from this list. Respond in the language used in the meeting. "
            f"Structure: **Client Name:** [Must match spelling in {attendee_names} if possible] "
            f"**Summary:** [Summary] **Business Plan:** [Plan]"
        )
        
        response = model.generate_content([file_upload, prompt])
        result_text = response.text

        logger.info(f"GEN GENERATED ANALYSIS:\n{result_text}")

        # 4. Name Fallback (If email lookup failed)
        if not contact_id:
            for line in result_text.split('\n'):
                if "**Client Name:**" in line:
                    detected_name = line.split(":**")[-1].strip()
                    contact_id = find_contact_in_ghl(detected_name)
                    break

        # 5. UPLOAD (With Real Logging)
        target = contact_id or GHL_CATCH_ALL_ID
        if target:
            upload_ghl_note(target, result_text)
            if not contact_id: logger.warning("Using Catch-All because specific contact was not found.")
        else:
            logger.error("No contact ID and no Catch-all ID found.")

    except Exception as e:
        logger.error(f"Logic Error: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path)
        if file_upload: genai.delete_file(file_upload.name)

# ... (rest of webhook code same as before) ...
