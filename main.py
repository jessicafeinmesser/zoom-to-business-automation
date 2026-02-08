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
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ------------------------------------------------------------------------------
# CONFIGURATION & LOGGING
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ZOOM_WEBHOOK_SECRET = os.getenv("ZOOM_WEBHOOK_SECRET")
GHL_API_KEY = os.getenv("GHL_API_KEY") # Your Zapier/Business Profile Key
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Your email
HOST_EMAILS = ["support@fullbookai.com"]

genai.configure(api_key=GOOGLE_API_KEY)

# V1 Base URL for the Zapier Key
GHL_BASE_URL = "https://api.gohighlevel.com/v1"

app = FastAPI()

# ------------------------------------------------------------------------------
# GHL HELPERS (V1)
# ------------------------------------------------------------------------------

def get_ghl_headers():
    return {
        "Authorization": f"Bearer {os.getenv('GHL_API_KEY', GHL_API_KEY)}",
        "Content-Type": "application/json"
    }

def get_ghl_contact_id(email: str) -> Optional[str]:
    """Search for contact by email in V1."""
    if not email or email in HOST_EMAILS:
        return None
    try:
        url = f"{GHL_BASE_URL}/contacts/"
        params = {"locationId": GHL_LOCATION_ID, "query": email}
        response = requests.get(url, headers=get_ghl_headers(), params=params)
        if response.status_code == 200:
            data = response.json()
            contacts = data.get("contacts", [])
            return contacts[0]["id"] if contacts else None
        return None
    except:
        return None

def create_ghl_note(contact_id: str, note_content: str):
    """Adds a note to a contact in V1."""
    try:
        url = f"{GHL_BASE_URL}/contacts/{contact_id}/notes"
        payload = {"body": note_content}
        requests.post(url, headers=get_ghl_headers(), json=payload).raise_for_status()
        logger.info(f"Note successfully uploaded to GHL contact {contact_id}")
    except Exception as e:
        logger.error(f"Failed to upload note to GHL: {e}")

# ------------------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------------------

def process_recording_logic(download_url: str, identified_email: str, download_token: str):
    temp_file_path = None
    file_upload = None

    try:
        logger.info(f"--- Starting Analysis. Target Email: {identified_email} ---")

        # 1. Download
        auth_url = f"{download_url}?access_token={download_token}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp_file_path = tmp.name
            with requests.get(auth_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=16384):
                    tmp.write(chunk)
        
        logger.info(f"Download complete. Size: {os.path.getsize(temp_file_path)} bytes.")

        # 2. Upload to Gemini
        file_upload = genai.upload_file(temp_file_path, mime_type="video/mp4")
        while file_upload.state.name == "PROCESSING":
            time.sleep(10)
            file_upload = genai.get_file(file_upload.name)
        
        if file_upload.state.name != "ACTIVE":
            logger.error("Gemini failed to process video.")
            return

        time.sleep(20) # Buffer

        # 3. Model & Safety
        available_names = [m.name for m in genai.list_models()]
        chosen_model = "models/gemini-flash-latest" if "models/gemini-flash-latest" in available_names else "models/gemini-1.5-flash"
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # 4. Generate
        logger.info(f"Generating AI analysis using {chosen_model}...")
        model = genai.GenerativeModel(model_name=chosen_model)
        prompt = (
            "Analyze this meeting recording carefully. Detect the language (Hebrew or English). "
            "Provide a Summary, Business Plan, and CRM Note in the detected language. "
        )

        response = model.generate_content([file_upload, prompt], safety_settings=safety_settings)
        
        if not response.text:
            logger.error("AI response empty.")
            return

        # --- FAIL-SAFE: ALWAYS LOG THE RESULT ---
        result_text = response.text
        logger.info("====================================================")
        logger.info(f"AI ANALYSIS RESULT FOR: {identified_email}")
        logger.info("\n" + result_text)
        logger.info("====================================================")

        # 5. GHL Upload
        contact_id = get_ghl_contact_id(identified_email)
        if contact_id:
            create_ghl_note(contact_id, result_text)
        else:
            logger.warning(f"Note not uploaded to CRM: No contact found for {identified_email}")

    except Exception as e:
        logger.error(f"Background Task Error: {e}")
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
            duration = obj.get("duration", 0) # Duration in minutes
            
            # Identify the client email
            # 1. Try registrant first
            email = obj.get("registrant_email")
            
            # 2. If no registrant, we use the host email as a placeholder for the log
            # but we won't skip anymore unless it's a tiny 1-minute test.
            if not email:
                email = obj.get("host_email")

            # --- NEW FAIL-SAFE LOGIC ---
            # If the meeting lasted more than 2 minutes, we assume it's a real meeting
            # and we process it regardless of whether we found a client email yet.
            if duration < 5:
                logger.info(f"Short meeting ({duration} min). Skipping.")
                return {"status": "skipped"}

            # Get MP4 URL
            download_url = next((f.get("download_url") for f in obj.get("recording_files", []) 
                                if f.get("file_type") == "MP4"), None)

            if download_url and download_token:
                background_tasks.add_task(process_recording_logic, download_url, email, download_token)
                logger.info(f"Queued analysis for meeting. Identified Email: {email}")
                return {"status": "queued"}

        return {"status": "ignored"}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error"}

@app.get("/")
def home():
    return {"status": "online"}
