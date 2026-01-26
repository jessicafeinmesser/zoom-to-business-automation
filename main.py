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

ZOOM_WEBHOOK_SECRET = os.getenv("ZOOM_WEBHOOK_SECRET")
GHL_API_KEY = os.getenv("GHL_API_KEY")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Exclusion List (Host/Personal Emails)
EXCLUDED_EMAILS = ["support@fullbookai.com"]

genai.configure(api_key=GOOGLE_API_KEY)
GHL_BASE_URL = "https://services.leadconnectorhq.com"

app = FastAPI()

# ------------------------------------------------------------------------------
# GHL HELPERS
# ------------------------------------------------------------------------------

def get_ghl_headers():
    # It is critical that GHL_API_KEY is the 'Location API Key' from Business Profile
    return {
        "Authorization": f"Bearer {os.getenv('GHL_API_KEY', GHL_API_KEY)}",
        "Version": "2021-07-28",
        "Content-Type": "application/json"
    }

def find_client_email_from_ghl(zoom_id: str) -> Optional[str]:
    """Searches GHL calendar events to find the client booked for this Zoom ID."""
    try:
        now = datetime.utcnow()
        start_time = int((now - timedelta(hours=3)).timestamp() * 1000)
        end_time = int((now + timedelta(hours=3)).timestamp() * 1000)

        url = f"{GHL_BASE_URL}/calendars/events"
        params = {
            "locationId": GHL_LOCATION_ID,
            "startTime": start_time,
            "endTime": end_time
        }
        
        response = requests.get(url, headers=get_ghl_headers(), params=params)
        response.raise_for_status()
        events = response.json().get("events", [])

        for event in events:
            address = event.get("address", "")
            if zoom_id in address:
                contact_id = event.get("contactId")
                if contact_id:
                    c_url = f"{GHL_BASE_URL}/contacts/{contact_id}"
                    c_resp = requests.get(c_url, headers=get_ghl_headers())
                    c_resp.raise_for_status()
                    return c_resp.json().get("contact", {}).get("email")
        return None
    except Exception as e:
        logger.error(f"Error looking up GHL event for Zoom ID {zoom_id}: {e}")
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
        logger.error(f"GHL contact search error: {e}")
        return None

def create_ghl_note(contact_id: str, note_content: str):
    try:
        url = f"{GHL_BASE_URL}/contacts/{contact_id}/notes"
        payload = {"body": note_content}
        requests.post(url, headers=get_ghl_headers(), json=payload).raise_for_status()
        logger.info(f"Note successfully added to GHL contact {contact_id}")
    except Exception as e:
        logger.error(f"Failed to create GHL note: {e}")

# ------------------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------------------

def process_recording_logic(download_url: str, client_email: str, download_token: str):
    temp_file_path = None
    file_upload = None

    try:
        logger.info(f"--- Starting Processing for: {client_email} ---")

        # 1. Download Video
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
            time.sleep(10)
            file_upload = genai.get_file(file_upload.name)
        
        if file_upload.state.name != "ACTIVE":
            logger.error("Gemini upload failed to activate.")
            return

        time.sleep(20) # AI Indexing Buffer

        # 3. Model & Safety Setup
        available_names = [m.name for m in genai.list_models()]
        chosen_model = "models/gemini-flash-latest" if "models/gemini-flash-latest" in available_names else "models/gemini-1.5-flash"
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # 4. Generate Analysis
        logger.info(f"Generating AI analysis using {chosen_model}...")
        model = genai.GenerativeModel(model_name=chosen_model)
        prompt = (
            "Analyze this meeting recording carefully. Detect the language (Hebrew or English). "
            "Provide a Summary, Business Plan, and CRM Note in the detected language. "
        )

        response = model.generate_content([file_upload, prompt], safety_settings=safety_settings)
        
        if not response.text:
            logger.error(f"AI returned empty result. Feedback: {response.prompt_feedback}")
            return

        # --- MANDATORY LOGGING OF ANALYSIS ---
        # This ensures you see the result even if GHL fails (401 error)
        result_text = response.text
        logger.info("====================================================")
        logger.info(f"AI ANALYSIS COMPLETED FOR {client_email}:")
        logger.info("\n" + result_text)
        logger.info("====================================================")

        # 5. GHL Integration
        contact_id = get_ghl_contact(client_email)
        if contact_id:
            create_ghl_note(contact_id, result_text)
        else:
            logger.warning(f"GHL Upload Skipped: Contact {client_email} not found or GHL API Error.")

    except Exception as e:
        logger.error(f"Background Process failed: {e}")
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
            zoom_id = str(obj.get("id"))
            
            # Identify Client
            client_email = obj.get("registrant_email")
            if not client_email:
                client_email = find_client_email_from_ghl(zoom_id)
            
            if not client_email:
                client_email = obj.get("host_email")

            # Exclusion Check
            if client_email in EXCLUDED_EMAILS:
                logger.info(f"Skipping: {client_email} is an excluded host email.")
                return {"status": "skipped"}

            # Get MP4 URL
            download_url = next((f.get("download_url") for f in obj.get("recording_files", []) 
                                if f.get("file_type") == "MP4"), None)

            if download_url and download_token:
                background_tasks.add_task(process_recording_logic, download_url, client_email, download_token)
                logger.info(f"Webhook Valid. Processing queued for client: {client_email}")
                return {"status": "queued"}

        return {"status": "ignored"}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error"}

@app.get("/")
def home():
    return {"status": "online"}
