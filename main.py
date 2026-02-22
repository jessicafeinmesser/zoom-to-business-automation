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

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
# Force the library to use the stable V1 API
genai.configure(api_key=GOOGLE_API_KEY)

HOST_EMAILS = ["support@fullbookai.com", "ofer.rapaport@gmail.com"]
PROCESSED_UUIDS: Set[str] = set()

app = FastAPI()

# ------------------------------------------------------------------------------
# HELPERS
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
    
    # Proper Double-Encoding for UUIDs
    encoded_uuid = urllib.parse.quote(urllib.parse.quote(meeting_uuid, safe=''), safe='')
    url = f"https://api.zoom.us/v2/report/meetings/{encoded_uuid}/participants"
    
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if response.status_code != 200:
            logger.warning(f"Zoom API Error: {data.get('message')}. Did you add the 'report:read' scope?")
            return None

        participants = data.get("participants", [])
        logger.info(f"--- ZOOM ATTENDEES FOR {meeting_uuid} ---")
        for p in participants:
            email = p.get("user_email")
            name = p.get("name")
            logger.info(f"Attendee: {name} | Email: {email}")
            if email and email.lower() not in [e.lower() for e in HOST_EMAILS]:
                return email.lower()
        return None
    except Exception as e:
        logger.error(f"Zoom Participant Fetch Error: {e}")
        return None

def find_client_by_appointment(zoom_id: str) -> Optional[str]:
    try:
        now = datetime.utcnow()
        start = int((now - timedelta(hours=24)).timestamp() * 1000)
        end = int((now + timedelta(hours=24)).timestamp() * 1000)
        url = f"{GHL_BASE_URL}/appointments/"
        params = {"locationId": GHL_LOCATION_ID, "startDate": start, "endDate": end}
        response = requests.get(url, headers={"Authorization": f"Bearer {GHL_API_KEY}"}, params=params)
        appts = response.json().get("appointments", [])
        clean_id = str(zoom_id).replace("-", "")
        for appt in appts:
            blob = (str(appt.get("location", "")) + str(appt.get("title", ""))).replace("-", "")
            if clean_id in blob: return appt.get("contactId")
        return None
    except: return None

# ------------------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------------------

def process_recording_logic(download_url: str, zoom_id: str, zoom_uuid: str, download_token: str):
    temp_file_path = None
    file_upload = None

    try:
        logger.info(f"--- STARTING ANALYSIS FOR MEETING {zoom_id} ---")

        # 1. FIND CONTACT
        contact_id = None
        guest_email = get_guest_email_from_zoom(zoom_uuid)
        if guest_email:
            url = f"{GHL_BASE_URL}/contacts/"
            params = {"locationId": GHL_LOCATION_ID, "query": guest_email}
            resp = requests.get(url, headers={"Authorization": f"Bearer {GHL_API_KEY}"}, params=params)
            contacts = resp.json().get("contacts", [])
            if contacts: 
                contact_id = contacts[0]["id"]
                logger.info(f"Matched contact via Email: {guest_email}")
        
        if not contact_id:
            logger.info("Email lookup failed. Trying GHL Appointments...")
            contact_id = find_client_by_appointment(zoom_id)

        # 2. DOWNLOAD
        auth_url = f"{download_url}?access_token={download_token}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp_file_path = tmp.name
            with requests.get(auth_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=16384): tmp.write(chunk)
        
        # 3. GEMINI
        file_upload = genai.upload_file(temp_file_path, mime_type="video/mp4")
        while file_upload.state.name == "PROCESSING":
            time.sleep(5)
            file_upload = genai.get_file(file_upload.name)
        
        time.sleep(10) # Indexing buffer

        # MODEL PICKER (Based on your visible list)
        available_models = [m.name for m in genai.list_models()]
        
        if "models/gemini-2.0-flash" in available_models:
            target_model = "models/gemini-2.0-flash"
        elif "models/gemini-flash-latest" in available_models:
            target_model = "models/gemini-flash-latest"
        else:
            # Last fallback
            target_model = "models/gemini-1.5-flash"

        logger.info(f"Using identified model: {target_model}")
        model = genai.GenerativeModel(model_name=target_model)
        
        prompt = (
            "Analyze this recording. Detect language (Hebrew or English). Respond ONLY in that language. "
            "Structure: **Client Name:** [Name] **Summary:** [Summary] **Business Plan:** [Plan]"
        )
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content([file_upload, prompt], safety_settings=safety_settings)
        result_text = response.text

        # ALWAYS LOG RESULT
        print("\n" + "="*60 + f"\nANALYSIS FOR {zoom_id}\n" + "-"*60 + f"\n{result_text}\n" + "="*60 + "\n")

        # 4. NAME FALLBACK & UPLOAD
        if not contact_id:
            logger.info("No contact matched. Trying AI Name Lookup...")
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
            logger.info(f"SUCCESS: Note uploaded to Contact {contact_id}")
        else:
            logger.error("GHL ERROR: No contact found. FULL PLAN IS LOGGED ABOVE.")

    except Exception as e:
        logger.error(f"Critical Processing Error: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path)
        if file_upload: genai.delete_file(file_upload.name)

# ------------------------------------------------------------------------------
# WEBHOOK
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
