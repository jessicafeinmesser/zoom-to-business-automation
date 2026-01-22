import os
import logging
import hmac
import hashlib
import json
import time
import tempfile
import requests
from typing import Optional, Dict, Any
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
import google.generativeai as genai

# ------------------------------------------------------------------------------
# CONFIGURATION & LOGGING
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment Variables
ZOOM_WEBHOOK_SECRET = os.getenv("ZOOM_WEBHOOK_SECRET")
GHL_API_KEY = os.getenv("GHL_API_KEY")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Exclusion List (Your own emails)
EXCLUDED_EMAILS = ["support@fullbookai.com"]

genai.configure(api_key=GOOGLE_API_KEY)
GHL_BASE_URL = "https://services.leadconnectorhq.com"

app = FastAPI()

# ------------------------------------------------------------------------------
# GHL HELPERS
# ------------------------------------------------------------------------------

def get_ghl_headers():
    return {
        "Authorization": f"Bearer {os.getenv('GHL_API_KEY', GHL_API_KEY)}",
        "Version": "2021-07-28",
        "Content-Type": "application/json"
    }

def find_client_email_from_ghl(zoom_id: str) -> Optional[str]:
    """
    Searches GHL appointments to find which client is booked for this Zoom ID.
    """
    try:
        # Search recent appointments in this location
        url = f"{GHL_BASE_URL}/appointments/"
        params = {
            "locationId": GHL_LOCATION_ID,
            "includeUpcoming": "true" 
        }
        
        response = requests.get(url, headers=get_ghl_headers(), params=params)
        response.raise_for_status()
        appointments = response.json().get("appointments", [])

        for appt in appointments:
            location = appt.get("location", "")
            # Check if this Zoom ID appears in the appointment's location field (the Zoom link)
            if zoom_id in location:
                # Get the contact associated with this appointment
                contact_id = appt.get("contactId")
                if contact_id:
                    # Look up contact to get their email
                    c_url = f"{GHL_BASE_URL}/contacts/{contact_id}"
                    c_resp = requests.get(c_url, headers=get_ghl_headers())
                    c_resp.raise_for_status()
                    return c_resp.json().get("contact", {}).get("email")
        
        return None
    except Exception as e:
        logger.error(f"Error looking up GHL appointment for Zoom ID {zoom_id}: {e}")
        return None

def get_ghl_contact(email: str) -> Optional[str]:
    try:
        url = f"{GHL_BASE_URL}/contacts/"
        params = {"locationId": GHL_LOCATION_ID, "query": email, "limit": 1}
        response = requests.get(url, headers=get_ghl_headers(), params=params)
        response.raise_for_status()
        contacts = response.json().get("contacts", [])
        return contacts[0]["id"] if contacts else None
    except Exception as e:
        logger.error(f"GHL search error: {e}")
        return None

def create_ghl_note(contact_id: str, note_content: str):
    try:
        url = f"{GHL_BASE_URL}/contacts/{contact_id}/notes"
        payload = {"body": note_content}
        requests.post(url, headers=get_ghl_headers(), json=payload).raise_for_status()
        logger.info(f"Note added to contact {contact_id}")
    except Exception as e:
        logger.error(f"Failed to add GHL note: {e}")

# ------------------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------------------

def process_recording_logic(download_url: str, client_email: str, download_token: str):
    temp_file_path = None
    file_upload = None

    try:
        logger.info(f"Starting analysis for client: {client_email}")

        # 1. Download
        auth_url = f"{download_url}?access_token={download_token}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp_file_path = tmp.name
            with requests.get(auth_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=16384):
                    tmp.write(chunk)
        
        # 2. Upload to Gemini
        file_upload = genai.upload_file(temp_file_path, mime_type="video/mp4")
        while file_upload.state.name == "PROCESSING":
            time.sleep(5)
            file_upload = genai.get_file(file_upload.name)
        
        if file_upload.state.name != "ACTIVE":
            return

        time.sleep(10)

        # 3. Analyze
        available_names = [m.name for m in genai.list_models()]
        model_name = "models/gemini-flash-latest" if "models/gemini-flash-latest" in available_names else "models/gemini-1.5-flash"
        model = genai.GenerativeModel(model_name)
        
        prompt = (
            "Analyze this recording. Detect the language (Hebrew or English). "
            "Provide a Summary, Business Plan, and CRM Note in the detected language."
        )

        response = model.generate_content([file_upload, prompt])
        
        # 4. Save to GHL
        contact_id = get_ghl_contact(client_email)
        if contact_id:
            create_ghl_note(contact_id, response.text)
            logger.info(f"COMPLETED: Analysis for {client_email} uploaded to GHL.")
        else:
            logger.warning(f"Analysis done for {client_email}, but contact not found in GHL.")

    except Exception as e:
        logger.error(f"Process failed: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if file_upload:
            try:
                genai.delete_file(file_upload.name)
            except:
                pass

# ------------------------------------------------------------------------------
# WEBHOOK ENDPOINT
# ------------------------------------------------------------------------------

@app.post("/zoom-webhook")
async def zoom_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        event = data.get("event")
        payload = data.get("payload", {})

        if event == "endpoint.url_validation":
            token = payload.get("plainToken")
            hashed = hmac.new(ZOOM_WEBHOOK_SECRET.encode(), token.encode(), hashlib.sha256).hexdigest()
            return {"plainToken": token, "encryptedToken": hashed}

        if event == "recording.completed":
            download_token = data.get("download_token")
            obj = payload.get("object", {})
            zoom_id = str(obj.get("id")) # Meeting ID (e.g., 8472938472)
            
            # Step A: Check for registrant email first
            client_email = obj.get("registrant_email")
            
            # Step B: If no registrant (standard GHL booking), lookup in GHL Calendar
            if not client_email:
                logger.info(f"No registrant email found. Searching GHL for Zoom ID: {zoom_id}")
                client_email = find_client_email_from_ghl(zoom_id)

            # Step C: Safety Check - Don't process if it's just you
            if not client_email or client_email in EXCLUDED_EMAILS:
                logger.info("Internal meeting or host-only meeting detected. Skipping.")
                return {"status": "skipped"}

            # Get MP4 URL
            download_url = next((f.get("download_url") for f in obj.get("recording_files", []) 
                                if f.get("file_type") == "MP4"), None)

            if download_url and download_token:
                background_tasks.add_task(process_recording_logic, download_url, client_email, download_token)
                logger.info(f"Success! Identified client {client_email}. Analysis queued.")
                return {"status": "queued"}

        return {"status": "ignored"}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error"}

@app.get("/")
def home():
    return {"status": "online"}
