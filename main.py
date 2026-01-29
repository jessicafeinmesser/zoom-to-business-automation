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

# Emails that belong to you (the host)
HOST_EMAILS = ["support@fullbookai.com"]

genai.configure(api_key=GOOGLE_API_KEY)

# UPDATED: Using the V1 Base URL for the Zapier Key
GHL_BASE_URL = "https://api.gohighlevel.com/v1"

app = FastAPI()

# ------------------------------------------------------------------------------
# GHL HELPERS (V1 COMPATIBLE)
# ------------------------------------------------------------------------------

def get_ghl_headers():
    """V1 API Headers for the Zapier Key."""
    return {
        "Authorization": f"Bearer {os.getenv('GHL_API_KEY', GHL_API_KEY)}",
        "Content-Type": "application/json"
    }

def find_client_email_from_ghl(zoom_id: str) -> Optional[str]:
    """Finds client email using the V1 Appointments API."""
    try:
        # V1 Appointment list
        url = f"{GHL_BASE_URL}/appointments/"
        params = {"locationId": GHL_LOCATION_ID}
        
        response = requests.get(url, headers=get_ghl_headers(), params=params)
        
        if response.status_code == 401:
            logger.error("GHL V1 AUTH ERROR: 401 Unauthorized. Double check your API Key in Render.")
            return None

        response.raise_for_status()
        # V1 returns a list of appointments
        appointments = response.json() if isinstance(response.json(), list) else response.json().get("appointments", [])

        for appt in appointments:
            # Check location/address for the Zoom ID
            search_area = str(appt.get("location", "")) + str(appt.get("address", ""))
            if zoom_id in search_area:
                contact_id = appt.get("contactId")
                if contact_id:
                    # Look up contact in V1
                    c_url = f"{GHL_BASE_URL}/contacts/{contact_id}"
                    c_resp = requests.get(c_url, headers=get_ghl_headers())
                    if c_resp.status_code == 200:
                        return c_resp.json().get("contact", {}).get("email")
        return None
    except Exception as e:
        logger.error(f"GHL V1 Appointment Lookup Exception: {e}")
        return None

def get_ghl_contact_id(email: str) -> Optional[str]:
    """Search for contact by email in V1."""
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
        logger.error(f"Failed to upload note to GHL V1: {e}")

# ------------------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------------------

def process_recording_logic(download_url: str, client_email: str, download_token: str):
    temp_file_path = None
    file_upload = None

    try:
        display_email = client_email if client_email else "Client_Email_Not_Detected"
        logger.info(f"--- Starting Analysis for: {display_email} ---")

        # 1. Download
        auth_url = f"{download_url}?access_token={download_token}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp_file_path = tmp.name
            with requests.get(auth_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=16384):
                    tmp.write(chunk)
        
        # 2. Gemini Upload
        file_upload = genai.upload_file(temp_file_path, mime_type="video/mp4")
        while file_upload.state.name == "PROCESSING":
            time.sleep(10)
            file_upload = genai.get_file(file_upload.name)
        
        if file_upload.state.name != "ACTIVE":
            logger.error("Gemini failed to process video.")
            return

        time.sleep(20) # AI Indexing Buffer

        # 3. Model Setup
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

        # --- FAIL-SAFE: LOG THE RESULT ---
        result_text = response.text
        logger.info("====================================================")
        logger.info(f"AI ANALYSIS COMPLETE FOR CLIENT: {display_email}")
        logger.info("\n" + result_text)
        logger.info("====================================================")

        # 5. GHL Upload
        if client_email and client_email not in HOST_EMAILS:
            contact_id = get_ghl_contact_id(client_email)
            if contact_id:
                create_ghl_note(contact_id, result_text)
            else:
                logger.warning(f"Note not uploaded: Contact {client_email} not found in GHL.")
        else:
            logger.info("Note not uploaded: No client identified or meeting was with self.")

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
            zoom_id = str(obj.get("id"))
            
            # Identify the client
            client_email = obj.get("registrant_email")
            if not client_email:
                client_email = find_client_email_from_ghl(zoom_id)
            
            # --- FAIL-SAFE ---
            # Even if GHL lookup fails, we process the video if there's any chance it's a client.
            # We only skip if the ONLY email Zoom provides is your support email.
            if not client_email:
                client_email = obj.get("host_email")

            if client_email in HOST_EMAILS and not obj.get("registrant_email"):
                logger.info("Host-only meeting detected. Skipping.")
                return {"status": "skipped"}

            # Get MP4 URL
            download_url = next((f.get("download_url") for f in obj.get("recording_files", []) 
                                if f.get("file_type") == "MP4"), None)

            if download_url and download_token:
                background_tasks.add_task(process_recording_logic, download_url, client_email, download_token)
                logger.info(f"Queued analysis for: {client_email}")
                return {"status": "queued"}

        return {"status": "ignored"}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"status": "error"}

@app.get("/")
def home():
    return {"status": "online"}
