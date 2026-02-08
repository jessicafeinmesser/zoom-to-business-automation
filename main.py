import os
import logging
import hmac
import hashlib
import json
import time
import tempfile
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ------------------------------------------------------------------------------
# CONFIGURATION & LOGGING
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sensitive Keys - Set these in Render Environment Variables
ZOOM_WEBHOOK_SECRET = os.getenv("ZOOM_WEBHOOK_SECRET")
GHL_API_KEY = os.getenv("GHL_API_KEY") 
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Emails to ignore (your own)
HOST_EMAILS = ["support@fullbookai.com"]

genai.configure(api_key=GOOGLE_API_KEY)

# V1 Base URL for your Zapier/Business Profile Key
GHL_BASE_URL = "https://api.gohighlevel.com/v1"

app = FastAPI()

# ------------------------------------------------------------------------------
# GHL HELPERS (V1 COMPATIBLE)
# ------------------------------------------------------------------------------

def get_ghl_headers():
    return {
        "Authorization": f"Bearer {os.getenv('GHL_API_KEY', GHL_API_KEY)}",
        "Content-Type": "application/json"
    }

def find_client_by_appointment(zoom_id: str) -> Optional[Dict]:
    """Search GHL V1 appointments for a matching Zoom Meeting ID."""
    try:
        now = datetime.utcnow()
        start_date = int((now - timedelta(hours=12)).timestamp() * 1000)
        end_date = int((now + timedelta(hours=12)).timestamp() * 1000)

        url = f"{GHL_BASE_URL}/appointments/"
        params = {"locationId": GHL_LOCATION_ID, "startDate": start_date, "endDate": end_date}
        
        response = requests.get(url, headers=get_ghl_headers(), params=params)
        if response.status_code != 200: return None

        appts = response.json()
        if not isinstance(appts, list): appts = appts.get("appointments", [])

        for appt in appts:
            loc = str(appt.get("location", "")) + str(appt.get("address", ""))
            if str(zoom_id) in loc:
                contact_id = appt.get("contactId")
                if contact_id:
                    c_url = f"{GHL_BASE_URL}/contacts/{contact_id}"
                    c_resp = requests.get(c_url, headers=get_ghl_headers())
                    if c_resp.status_code == 200:
                        contact = c_resp.json().get("contact", {})
                        return {"id": contact.get("id"), "email": contact.get("email"), "name": contact.get("firstName")}
        return None
    except Exception as e:
        logger.error(f"GHL Appointment Search Error: {e}")
        return None

def find_contact_by_name(name: str, zoom_id: str) -> Optional[str]:
    """Search GHL by Name and narrow down using the Zoom ID if multiple found."""
    if not name or len(name) < 2: return None
    try:
        url = f"{GHL_BASE_URL}/contacts/"
        params = {"locationId": GHL_LOCATION_ID, "query": name}
        response = requests.get(url, headers=get_ghl_headers(), params=params)
        if response.status_code != 200: return None
        
        contacts = response.json().get("contacts", [])
        if not contacts: return None
        if len(contacts) == 1: return contacts[0]["id"]

        # Multiple found - check which one has a meeting with this Zoom ID
        logger.info(f"Multiple contacts found for '{name}'. Cross-referencing Zoom ID...")
        appt_match = find_client_by_appointment(zoom_id)
        if appt_match:
            for c in contacts:
                if c["id"] == appt_match["id"]: return c["id"]

        return contacts[0]["id"] # Final fallback: first match
    except Exception as e:
        logger.error(f"GHL Name Match Error: {e}")
        return None

def create_ghl_note(contact_id: str, note_content: str):
    """Adds note to contact in V1."""
    try:
        url = f"{GHL_BASE_URL}/contacts/{contact_id}/notes"
        requests.post(url, headers=get_ghl_headers(), json={"body": note_content}).raise_for_status()
        logger.info(f"Successfully uploaded analysis to Contact ID: {contact_id}")
    except Exception as e:
        logger.error(f"GHL Note Creation Failed: {e}")

# ------------------------------------------------------------------------------
# CORE PROCESSING LOGIC
# ------------------------------------------------------------------------------

def process_recording_logic(download_url: str, zoom_id: str, download_token: str, initial_email: str):
    temp_file_path = None
    file_upload = None

    try:
        logger.info(f"--- Processing Zoom Meeting: {zoom_id} ---")

        # 1. Identify Client (Level 1: Appointment Match)
        client_data = find_client_by_appointment(zoom_id)
        contact_id = client_data["id"] if client_data else None
        target_email = client_data["email"] if client_data else initial_email

        # 2. Download Media
        auth_url = f"{download_url}?access_token={download_token}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp_file_path = tmp.name
            with requests.get(auth_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=16384): tmp.write(chunk)
        
        # 3. Gemini Upload & Process
        file_upload = genai.upload_file(temp_file_path, mime_type="video/mp4")
        while file_upload.state.name == "PROCESSING":
            time.sleep(10)
            file_upload = genai.get_file(file_upload.name)
        
        time.sleep(20) # Essential buffer for video indexing

        # 4. Model Selection & Safety
        available_names = [m.name for m in genai.list_models()]
        chosen_model = "models/gemini-flash-latest" if "models/gemini-flash-latest" in available_names else "models/gemini-1.5-flash"
        
        # Turn off safety filters to prevent empty responses
        safety_settings = {
            cat: HarmBlockThreshold.BLOCK_NONE for cat in [
                HarmCategory.HARM_CATEGORY_HARASSMENT, HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT
            ]
        }

        # 5. Generate with Client Detection
        model = genai.GenerativeModel(model_name=chosen_model)
        prompt = (
            "Analyze this recording. Detect the language (Hebrew or English).\n"
            "Respond ONLY in that language. Structure the response exactly like this:\n"
            "**Client Name:** [Extract client name from audio]\n"
            "**Summary:** [Concise summary]\n"
            "**Business Plan:** [Detailed actionable plan]\n"
            "**CRM Note:** [Short context note]"
        )

        response = model.generate_content([file_upload, prompt], safety_settings=safety_settings)
        result_text = response.text

        # ALWAYS LOG THE RESULT (Fail-safe)
        logger.info("====================================================")
        logger.info(result_text)
        logger.info("====================================================")

        # 6. Identify Client (Level 2: AI Name Extraction)
        if not contact_id:
            logger.info("Email match failed. Searching GHL by AI detected name...")
            for line in result_text.split('\n'):
                if "**Client Name:**" in line:
                    detected_name = line.replace("**Client Name:**", "").strip()
                    contact_id = find_contact_by_name(detected_name, zoom_id)
                    break

        # 7. Final GHL Upload
        if contact_id:
            create_ghl_note(contact_id, result_text)
        else:
            logger.warning(f"Could not link this meeting to a GHL contact for {target_email}.")

    except Exception as e:
        logger.error(f"Process Error: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path)
        if file_upload: genai.delete_file(file_upload.name)

# ------------------------------------------------------------------------------
# API ENDPOINT
# ------------------------------------------------------------------------------

@app.post("/zoom-webhook")
async def zoom_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        event = data.get("event")
        payload = data.get("payload", {})

        # Zoom Handshake
        if event == "endpoint.url_validation":
            token = payload.get("plainToken")
            hashed = hmac.new(ZOOM_WEBHOOK_SECRET.encode(), token.encode(), hashlib.sha256).hexdigest()
            return {"plainToken": token, "encryptedToken": hashed}

        # Recording Completed
        if event == "recording.completed":
            download_token = data.get("download_token")
            obj = payload.get("object", {})
            zoom_id = str(obj.get("id"))
            
            # Initial ID Check
            email = obj.get("registrant_email") or obj.get("host_email")

            # Only process if > 5 minutes (ignores tiny test calls)
            if obj.get("duration", 0) < 2:
                logger.info(f"Meeting {zoom_id} is too short. Skipping.")
                return {"status": "skipped"}

            # Get MP4 URL
            download_url = next((f.get("download_url") for f in obj.get("recording_files", []) 
                                if f.get("file_type") == "MP4"), None)

            if download_url and download_token:
                background_tasks.add_task(process_recording_logic, download_url, zoom_id, download_token, email)
                logger.info(f"Queued analysis for Zoom ID: {zoom_id}")
                return {"status": "queued"}

        return {"status": "ignored"}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error"}

@app.get("/")
def home(): return {"status": "online"}
