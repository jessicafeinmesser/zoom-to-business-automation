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
GHL_API_KEY = os.getenv("GHL_API_KEY")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Add your personal/support emails here to skip processing them
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
    """Finds client email by searching for the Zoom Link in GHL Appointments."""
    try:
        url = f"{GHL_BASE_URL}/appointments/"
        params = {"locationId": GHL_LOCATION_ID, "includeUpcoming": "true"}
        response = requests.get(url, headers=get_ghl_headers(), params=params)
        response.raise_for_status()
        appointments = response.json().get("appointments", [])

        for appt in appointments:
            location = appt.get("location", "")
            if zoom_id in location:
                contact_id = appt.get("contactId")
                if contact_id:
                    c_url = f"{GHL_BASE_URL}/contacts/{contact_id}"
                    c_resp = requests.get(c_url, headers=get_ghl_headers())
                    c_resp.raise_for_status()
                    return c_resp.json().get("contact", {}).get("email")
        return None
    except Exception as e:
        logger.error(f"GHL Appointment Lookup Error: {e}")
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
    """Post analysis as a note to the contact with safety checks."""
    try:
        if not note_content or len(note_content.strip()) < 10:
            logger.warning("Note content is too short or empty. Skipping GHL upload.")
            return

        url = f"{GHL_BASE_URL}/contacts/{contact_id}/notes"
        
        # Ensure we are sending valid JSON
        payload = {"body": note_content}
        
        response = requests.post(url, headers=get_ghl_headers(), json=payload)
        
        if response.status_code == 401:
            logger.error("GHL ERROR: 401 Unauthorized. Please refresh your GHL API Key in Render.")
            return

        response.raise_for_status()
        logger.info(f"Successfully added Business Plan note to contact {contact_id}")
    except Exception as e:
        logger.error(f"GHL Note Creation Failed: {e}")

# ------------------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------------------

def process_recording_logic(download_url: str, client_email: str, download_token: str):
    temp_file_path = None
    file_upload = None

    try:
        logger.info(f"--- Starting Analysis for {client_email} ---")

        # 1. Download Video
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
            logger.error(f"Gemini file state failed: {file_upload.state.name}")
            return

        # CRITICAL: Wait 20 seconds for the AI to "index" the video content
        logger.info("File active. Waiting 20s for internal indexing...")
        time.sleep(20)

        # 3. Dynamic Model Picker
        available_names = [m.name for m in genai.list_models()]
        if "models/gemini-flash-latest" in available_names:
            chosen_model = "models/gemini-flash-latest"
        elif "models/gemini-1.5-flash" in available_names:
            chosen_model = "models/gemini-1.5-flash"
        else:
            chosen_model = next((n for n in available_names if "flash" in n), "models/gemini-1.5-flash")

        # 4. Safety Settings (Prevents Empty Responses)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # 5. Generate Content
        logger.info(f"Requesting analysis from {chosen_model}...")
        model = genai.GenerativeModel(model_name=chosen_model)
        prompt = (
            "Analyze this meeting recording carefully. Detect the language (Hebrew or English).\n"
            "Provide:\n1. Summary\n2. Detailed Business Plan\n3. CRM Note\n"
            "Respond ONLY in the detected language."
        )

        response = model.generate_content(
            [file_upload, prompt],
            safety_settings=safety_settings
        )
        
        # Check for blocked response
        if not response.candidates or not response.candidates[0].content.parts:
            logger.error(f"AI blocked the response. Feedback: {response.prompt_feedback}")
            return

        result_text = response.text
        logger.info("AI Analysis completed.")

        # 6. Save to GHL
        contact_id = get_ghl_contact(client_email)
        if contact_id:
            create_ghl_note(contact_id, result_text)
            logger.info(f"Successfully uploaded note for {client_email}")
        else:
            logger.warning(f"Analysis complete, but {client_email} not found in GHL.")

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
                logger.info(f"Skipping: {client_email} is an excluded email.")
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
