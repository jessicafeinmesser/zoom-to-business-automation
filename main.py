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

# Safety net: Create a GHL contact called "Zoom Archive" and put its ID here
GHL_CATCH_ALL_CONTACT_ID = os.getenv("GHL_CATCH_ALL_CONTACT_ID") 

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# EXCLUSION LIST: People who are NOT the client
HOST_EMAILS = [
    "support@fullbookai.com", 
    "info@fullbookai.com", 
    "ofer.rapaport@gmail.com"
]

PROCESSED_UUIDS: Set[str] = set()
app = FastAPI()

# ------------------------------------------------------------------------------
# GHL & ZOOM HELPERS
# ------------------------------------------------------------------------------

def get_zoom_access_token():
    url = f"https://zoom.us/oauth/token?grant_type=account_credentials&account_id={ZOOM_ACCOUNT_ID}"
    try:
        response = requests.post(url, auth=(ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET))
        return response.json().get("access_token")
    except: return None

def find_client_by_appointment(zoom_id: str) -> Optional[str]:
    """
    SEARCH HIERARCHY #1: 
    Find the contact who is actually booked for this Zoom ID in GHL.
    """
    try:
        now = datetime.utcnow()
        # Search window: 24 hours before/after to catch all timezones
        start = int((now - timedelta(hours=24)).timestamp() * 1000)
        end = int((now + timedelta(hours=24)).timestamp() * 1000)
        
        url = f"{GHL_BASE_URL}/appointments/"
        params = {"locationId": GHL_LOCATION_ID, "startDate": start, "endDate": end}
        response = requests.get(url, headers={"Authorization": f"Bearer {GHL_API_KEY}"}, params=params)
        
        if response.status_code != 200: return None
        
        appts = response.json().get("appointments", [])
        clean_id = str(zoom_id).replace("-", "")

        for appt in appts:
            # Check location (where Zoom link usually lives), title, and description
            search_blob = (
                str(appt.get("location", "")) + 
                str(appt.get("title", "")) + 
                str(appt.get("description", "")) +
                str(appt.get("address", ""))
            ).replace("-", "")
            
            if clean_id in search_blob:
                contact_id = appt.get("contactId")
                logger.info(f"GOLDEN TICKET: Found Contact ID {contact_id} via GHL Appointment.")
                return contact_id
        return None
    except Exception as e:
        logger.error(f"Appointment search error: {e}")
        return None

def get_guest_email_from_zoom(meeting_uuid: str) -> Optional[str]:
    """SEARCH HIERARCHY #2: Zoom Participant Email."""
    token = get_zoom_access_token()
    if not token: return None
    # Double-encode to handle '+' and '/' in UUIDs
    encoded_uuid = urllib.parse.quote(urllib.parse.quote(meeting_uuid, safe=''), safe='')
    url = f"https://api.zoom.us/v2/report/meetings/{encoded_uuid}/participants"
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers)
        participants = response.json().get("participants", [])
        
        logger.info(f"--- ZOOM ATTENDEES FOR {meeting_uuid} ---")
        for p in participants:
            email = p.get("user_email", "")
            logger.info(f"Attendee: {p.get('name')} | Email: '{email}'")
            if email and email.lower() not in [e.lower() for e in HOST_EMAILS]:
                return email.lower()
        return None
    except: return None

def find_contact_in_ghl(query: str) -> Optional[str]:
    """SEARCH HIERARCHY #3: Fuzzy Name/Email Search."""
    if not query or len(query) < 2: return None
    try:
        url = f"{GHL_BASE_URL}/contacts/"
        params = {"locationId": GHL_LOCATION_ID, "query": query}
        resp = requests.get(url, headers={"Authorization": f"Bearer {GHL_API_KEY}"}, params=params)
        contacts = resp.json().get("contacts", [])
        return contacts[0]["id"] if contacts else None
    except: return None

# ------------------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------------------

def process_recording_logic(download_url: str, zoom_id: str, zoom_uuid: str, download_token: str):
    temp_file_path = None
    file_upload = None

    try:
        logger.info(f"--- STARTING ANALYSIS FOR MEETING {zoom_id} ---")
        contact_id = None

        # 1. HIERARCHY #1: APPOINTMENT MATCH
        contact_id = find_client_by_appointment(zoom_id)

        # 2. HIERARCHY #2: ZOOM EMAIL MATCH
        if not contact_id:
            guest_email = get_guest_email_from_zoom(zoom_uuid)
            if guest_email:
                contact_id = find_contact_in_ghl(guest_email)

        # 3. DOWNLOAD & GEMINI
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
        
        time.sleep(10) # Index buffer
        
        model = genai.GenerativeModel(model_name="models/gemini-flash-latest")
        # Prompt logic: Extracts name in both or either language to maximize GHL match
        prompt = (
            "Analyze recording. Detect language (Hebrew or English). "
            "Structure response exactly as follows: "
            "**Client Name:** [Extract name. If the person has a Hebrew name in the system, use Hebrew characters. If English, use English.] "
            "**Summary:** [Concise Summary] "
            "**Business Plan:** [Detailed actionable plan]"
        )
        
        response = model.generate_content([file_upload, prompt])
        result_text = response.text

        # ALWAYS LOG RESULT (Render Logs)
        logger.info("============================================================")
        logger.info(f"TRANSCRIPTION & PLAN:\n{result_text}")
        logger.info("============================================================")

        # 4. HIERARCHY #3: AI NAME MATCH (Fallback)
        if not contact_id:
            logger.info("No contact matched via appointment or email. Trying AI Name...")
            for line in result_text.split('\n'):
                if "**Client Name:**" in line:
                    detected_name = line.split(":**")[-1].strip()
                    contact_id = find_contact_in_ghl(detected_name)
                    # Try fuzzy first name match if full name fails
                    if not contact_id and " " in detected_name:
                        first = detected_name.split(" ")[0]
                        contact_id = find_contact_in_ghl(first)
                    break

        # 5. HIERARCHY #4: THE CATCH-ALL SAFETY NET
        final_target = contact_id or GHL_CATCH_ALL_CONTACT_ID
        
        if final_target:
            url = f"{GHL_BASE_URL}/contacts/{final_target}/notes"
            requests.post(url, headers={"Authorization": f"Bearer {GHL_API_KEY}"}, json={"body": result_text})
            if not contact_id:
                logger.warning(f"Note sent to CATCH-ALL ({GHL_CATCH_ALL_CONTACT_ID})")
            else:
                logger.info(f"SUCCESS: Uploaded to Contact {contact_id}")
        else:
            logger.error("GHL ERROR: No contact matched and no Catch-All ID set.")

    except Exception as e:
        logger.error(f"Processing Error: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path)
        if file_upload: genai.delete_file(file_upload.name)

# ------------------------------------------------------------------------------
# WEBHOOK & HOME
# ------------------------------------------------------------------------------

@app.post("/zoom-webhook")
async def zoom_webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    event = data.get("event")
    if event == "endpoint.url_validation":
        token = data.get("payload", {}).get("plainToken")
        hashed = hmac.new(ZOOM_WEBHOOK_SECRET.encode(), token.encode(), hashlib.sha256).hexdigest()
        return {"plainToken": token, "encryptedToken": hashed}
    if event == "recording.completed":
        payload = data.get("payload", {}).get("object", {})
        zoom_uuid = str(payload.get("uuid"))
        if zoom_uuid in PROCESSED_UUIDS: return {"status": "skipped"}
        if payload.get("duration", 0) < 2: return {"status": "too_short"}
        PROCESSED_UUIDS.add(zoom_uuid)
        files = payload.get("recording_files", [])
        mp4_file = next((f for f in files if f.get("file_type") == "MP4" and "speaker" in f.get("recording_type", "").lower()), None)
        if not mp4_file: mp4_file = next((f for f in files if f.get("file_type") == "MP4"), None)
        if mp4_file:
            background_tasks.add_task(process_recording_logic, mp4_file.get("download_url"), str(payload.get("id")), zoom_uuid, data.get("download_token"))
            return {"status": "queued"}
    return {"status": "ignored"}

@app.get("/")
def home(): return {"status": "online"}
