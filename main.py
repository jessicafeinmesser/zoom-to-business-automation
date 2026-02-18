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

# Credentials
ZOOM_ACCOUNT_ID = os.getenv("ZOOM_ACCOUNT_ID")
ZOOM_CLIENT_ID = os.getenv("ZOOM_CLIENT_ID")
ZOOM_CLIENT_SECRET = os.getenv("ZOOM_CLIENT_SECRET")
ZOOM_WEBHOOK_SECRET = os.getenv("ZOOM_WEBHOOK_SECRET")

GHL_API_KEY = os.getenv("GHL_API_KEY")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID")
GHL_BASE_URL = "https://api.gohighlevel.com/v1"

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Emails to ignore (your own team)
HOST_EMAILS = ["support@fullbookai.com", "ofer.rapaport@gmail.com"]

app = FastAPI()

# ------------------------------------------------------------------------------
# ZOOM API HELPERS (OAuth & Participants)
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
    """Calls Zoom API to find the participant who isn't the host."""
    token = get_zoom_access_token()
    if not token: return None

    # Double encode UUID if it starts with / or has //
    safe_uuid = meeting_uuid.replace("/", "%2F").replace("//", "%2F%2F")
    url = f"https://api.zoom.us/v2/past_meetings/{safe_uuid}/participants"
    
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers)
        if response.status_code != 200: return None

        participants = response.json().get("participants", [])
        for p in participants:
            email = p.get("user_email")
            if email and email.lower() not in [e.lower() for e in HOST_EMAILS]:
                return email.lower()
        return None
    except Exception as e:
        logger.error(f"Error fetching Zoom participants: {e}")
        return None

# ------------------------------------------------------------------------------
# GHL HELPERS (V1 API)
# ------------------------------------------------------------------------------

def get_ghl_headers():
    return {"Authorization": f"Bearer {GHL_API_KEY}", "Content-Type": "application/json"}

def find_contact_by_email(email: str) -> Optional[str]:
    """Search GHL for a contact by email address."""
    if not email: return None
    try:
        url = f"{GHL_BASE_URL}/contacts/"
        params = {"locationId": GHL_LOCATION_ID, "query": email}
        response = requests.get(url, headers=get_ghl_headers(), params=params)
        contacts = response.json().get("contacts", [])
        return contacts[0]["id"] if contacts else None
    except Exception as e:
        logger.error(f"GHL Email Search Error: {e}")
        return None

def find_client_by_appointment(zoom_id: str) -> Optional[str]:
    """Search GHL appointments where Zoom ID is in the location/description."""
    try:
        # Check window of 12 hours before/after
        now = datetime.utcnow()
        start = int((now - timedelta(hours=12)).timestamp() * 1000)
        end = int((now + timedelta(hours=12)).timestamp() * 1000)

        url = f"{GHL_BASE_URL}/appointments/"
        params = {"locationId": GHL_LOCATION_ID, "startDate": start, "endDate": end}
        
        response = requests.get(url, headers=get_ghl_headers(), params=params)
        appts = response.json().get("appointments", []) if response.status_code == 200 else []

        for appt in appts:
            # Check if zoom ID exists in any text field of the appointment
            loc_data = str(appt.get("location", "")) + str(appt.get("address", "")) + str(appt.get("title", ""))
            if str(zoom_id) in loc_data:
                return appt.get("contactId")
        return None
    except Exception as e:
        logger.error(f"GHL Appointment Match Error: {e}")
        return None

def find_contact_by_name(name: str) -> Optional[str]:
    """Search GHL by Name as a last resort fallback."""
    if not name or len(name) < 3: return None
    try:
        url = f"{GHL_BASE_URL}/contacts/"
        params = {"locationId": GHL_LOCATION_ID, "query": name}
        response = requests.get(url, headers=get_ghl_headers(), params=params)
        contacts = response.json().get("contacts", [])
        return contacts[0]["id"] if contacts else None
    except:
        return None

def create_ghl_note(contact_id: str, note_content: str):
    """Adds the AI analysis to the contact's notes."""
    try:
        url = f"{GHL_BASE_URL}/contacts/{contact_id}/notes"
        requests.post(url, headers=get_ghl_headers(), json={"body": note_content}).raise_for_status()
        logger.info(f"Successfully uploaded note to Contact: {contact_id}")
    except Exception as e:
        logger.error(f"GHL Note Creation Failed: {e}")

# ------------------------------------------------------------------------------
# CORE PROCESSING LOGIC
# ------------------------------------------------------------------------------

def process_recording_logic(download_url: str, zoom_id: str, zoom_uuid: str, download_token: str):
    temp_file_path = None
    file_upload = None

    try:
        logger.info(f"--- STARTING ANALYSIS FOR MEETING {zoom_id} ---")

        # 1. FIND THE CONTACT (The "Failproof" Waterfall)
        contact_id = None
        
        # Step A: Check Zoom Participants API (Highest Accuracy)
        guest_email = get_guest_email_from_zoom(zoom_uuid)
        if guest_email:
            logger.info(f"Guest email found via Zoom API: {guest_email}")
            contact_id = find_contact_by_email(guest_email)
        
        # Step B: Check GHL Appointments by Zoom ID
        if not contact_id:
            logger.info("Email check failed. Searching GHL Appointments...")
            contact_id = find_client_by_appointment(zoom_id)

        # 2. DOWNLOAD RECORDING
        auth_url = f"{download_url}?access_token={download_token}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp_file_path = tmp.name
            with requests.get(auth_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=16384): tmp.write(chunk)
        
        # 3. GEMINI PROCESSING
        file_upload = genai.upload_file(temp_file_path, mime_type="video/mp4")
        while file_upload.state.name == "PROCESSING":
            time.sleep(5)
            file_upload = genai.get_file(file_upload.name)
        
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        prompt = (
            "Analyze this recording. Detect the language (Hebrew or English).\n"
            "Respond ONLY in that language. Structure the response like this:\n"
            "**Client Name:** [Extract client name from audio]\n"
            "**Summary:** [Concise summary]\n"
            "**Business Plan:** [Detailed actionable plan]\n"
            "**Action Items:** [Bullet points]"
        )

        safety_settings = {cat: HarmBlockThreshold.BLOCK_NONE for cat in [
            HarmCategory.HARM_CATEGORY_HARASSMENT, HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT
        ]}

        response = model.generate_content([file_upload, prompt], safety_settings=safety_settings)
        result_text = response.text

        # 4. FALLBACK: FIND BY AI EXTRACTED NAME
        if not contact_id:
            logger.info("Contact ID still empty. Attempting search by extracted name...")
            for line in result_text.split('\n'):
                if "**Client Name:**" in line:
                    detected_name = line.replace("**Client Name:**", "").strip()
                    contact_id = find_contact_by_name(detected_name)
                    break

        # 5. UPLOAD TO GHL
        if contact_id:
            create_ghl_note(contact_id, result_text)
        else:
            # THE FAILSAFE: If no contact is found, log it clearly so you can copy/paste it manually
            logger.error("!!! NO GHL CONTACT FOUND !!!")
            logger.error(f"ANALYSIS FOR ZOOM {zoom_id}:\n{result_text}")

    except Exception as e:
        logger.error(f"Process Error: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path)
        if file_upload: genai.delete_file(file_upload.name)

# ------------------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------------------

@app.post("/zoom-webhook")
async def zoom_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        event = data.get("event")
        payload = data.get("payload", {})

        # URL Validation (Handshake)
        if event == "endpoint.url_validation":
            token = payload.get("plainToken")
            hashed = hmac.new(ZOOM_WEBHOOK_SECRET.encode(), token.encode(), hashlib.sha256).hexdigest()
            return {"plainToken": token, "encryptedToken": hashed}

        # Recording Finished
        if event == "recording.completed":
            obj = payload.get("object", {})
            # We need both the ID (for GHL) and UUID (for Zoom API)
            zoom_id = str(obj.get("id"))
            zoom_uuid = str(obj.get("uuid"))
            
            # Skip meetings shorter than 2 minutes
            if obj.get("duration", 0) < 2:
                return {"status": "too_short"}

            # Find the MP4 file
            download_url = next((f.get("download_url") for f in obj.get("recording_files", []) 
                                if f.get("file_type") == "MP4"), None)
            
            download_token = data.get("download_token")

            if download_url and download_token:
                background_tasks.add_task(process_recording_logic, download_url, zoom_id, zoom_uuid, download_token)
                return {"status": "queued"}

        return {"status": "ignored"}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error"}

@app.get("/")
def home(): return {"status": "online"}
