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
# CONFIGURATION & LOGGING
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sensitive Keys from Environment Variables
ZOOM_ACCOUNT_ID = os.getenv("ZOOM_ACCOUNT_ID")
ZOOM_CLIENT_ID = os.getenv("ZOOM_CLIENT_ID")
ZOOM_CLIENT_SECRET = os.getenv("ZOOM_CLIENT_SECRET")
ZOOM_WEBHOOK_SECRET = os.getenv("ZOOM_WEBHOOK_SECRET")

GHL_API_KEY = os.getenv("GHL_API_KEY")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID")
GHL_BASE_URL = "https://api.gohighlevel.com/v1"

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Emails to ignore (your team)
HOST_EMAILS = ["support@fullbookai.com", "ofer.rapaport@gmail.com"]

app = FastAPI()

# ------------------------------------------------------------------------------
# ZOOM API HELPERS
# ------------------------------------------------------------------------------

def get_zoom_access_token():
    """Gets a temporary access token using Server-to-Server OAuth."""
    url = f"https://zoom.us/oauth/token?grant_type=account_credentials&account_id={ZOOM_ACCOUNT_ID}"
    try:
        response = requests.post(url, auth=(ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET))
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as e:
        logger.error(f"Failed to get Zoom Access Token: {e}")
        return None

def get_guest_email_from_zoom(meeting_uuid: str) -> Optional[str]:
    """Calls Zoom API to find the participant who isn't a host."""
    token = get_zoom_access_token()
    if not token: return None

    # Double encode UUID for API safety
    safe_uuid = meeting_uuid.replace("/", "%2F").replace("//", "%2F%2F")
    url = f"https://api.zoom.us/v2/report/meetings/{safe_uuid}/participants"
    
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers)
        if response.status_code != 200: return None

        participants = response.json().get("participants", [])
        for p in participants:
            email = p.get("user_email")
            # --- LINE 59 FIX: Correctly checking exclusion list ---
            if email and email.lower() not in [e.lower() for e in HOST_EMAILS]:
                return email.lower()
        return None
    except Exception as e:
        logger.error(f"Error fetching Zoom participants: {e}")
        return None

# ------------------------------------------------------------------------------
# GHL HELPERS
# ------------------------------------------------------------------------------

def find_client_by_appointment(zoom_id: str) -> Optional[str]:
    """Search GHL appointments where Zoom ID is in the location/title."""
    try:
        now = datetime.utcnow()
        start = int((now - timedelta(hours=24)).timestamp() * 1000)
        end = int((now + timedelta(hours=24)).timestamp() * 1000)

        url = f"{GHL_BASE_URL}/appointments/"
        params = {"locationId": GHL_LOCATION_ID, "startDate": start, "endDate": end}
        
        headers = {"Authorization": f"Bearer {GHL_API_KEY}"}
        response = requests.get(url, headers=headers, params=params)
        appts = response.json().get("appointments", []) if response.status_code == 200 else []

        clean_id = str(zoom_id).replace("-", "")
        for appt in appts:
            search_blob = (str(appt.get("location", "")) + str(appt.get("title", ""))).replace("-", "")
            if clean_id in search_blob:
                return appt.get("contactId")
        return None
    except Exception as e:
        logger.error(f"GHL Appointment Search Error: {e}")
        return None

# ------------------------------------------------------------------------------
# CORE PROCESSING LOGIC
# ------------------------------------------------------------------------------

def process_recording_logic(download_url: str, zoom_id: str, zoom_uuid: str, download_token: str):
    temp_file_path = None
    file_upload = None

    try:
        logger.info(f"--- STARTING ANALYSIS FOR MEETING {zoom_id} ---")

        # 1. FIND THE CONTACT
        contact_id = None
        guest_email = get_guest_email_from_zoom(zoom_uuid)
        if guest_email:
            logger.info(f"Guest email identified: {guest_email}")
            url = f"{GHL_BASE_URL}/contacts/"
            params = {"locationId": GHL_LOCATION_ID, "query": guest_email}
            resp = requests.get(url, headers={"Authorization": f"Bearer {GHL_API_KEY}"}, params=params)
            contacts = resp.json().get("contacts", [])
            if contacts: contact_id = contacts[0]["id"]
        
        if not contact_id:
            logger.info("Email check failed. Falling back to Appointment search...")
            contact_id = find_client_by_appointment(zoom_id)

        # 2. DOWNLOAD & TRANSCRIPTION
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
        
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt = (
            "Analyze this recording. Detect language (Hebrew or English). Respond ONLY in that language. "
            "Structure: **Client Name:** [Name] **Summary:** [Summary] **Business Plan:** [Plan]"
        )
        response = model.generate_content([file_upload, prompt])
        result_text = response.text

        # 3. ALWAYS LOG FULL RESULT (Backup)
        print("\n" + "="*60)
        print(f"FULL ANALYSIS FOR {zoom_id}:\n{result_text}")
        print("="*60 + "\n")

        # 4. NAME FALLBACK & UPLOAD
        if not contact_id:
            for line in result_text.split('\n'):
                if "**Client Name:**" in line:
                    detected_name = line.split(":**")[-1].strip()
                    url = f"{GHL_BASE_URL}/contacts/"
                    params = {"locationId": GHL_LOCATION_ID, "query": detected_name}
                    resp = requests.get(url, headers={"Authorization": f"Bearer {GHL_API_KEY}"}, params=params)
                    contacts = resp.json().get("contacts", [])
                    if contacts: contact_id = contacts[0]["id"]
                    break

        if contact_id:
            url = f"{GHL_BASE_URL}/contacts/{contact_id}/notes"
            requests.post(url, headers={"Authorization": f"Bearer {GHL_API_KEY}"}, json={"body": result_text})
            logger.info(f"SUCCESS: Analysis uploaded to G
