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
from pydantic import BaseModel
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ------------------------------------------------------------------------------
# CONFIGURATION & ENVIRONMENT VARIABLES
# ------------------------------------------------------------------------------

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sensitive Keys (As requested)
ZOOM_WEBHOOK_SECRET = os.getenv("ZOOM_WEBHOOK_SECRET", "UR6GqxUNSj-rFvVuQqy9_w")
GHL_API_KEY = os.getenv("GHL_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJsb2NhdGlvbl9pZCI6InN4Uk9jUWlUMXlIaGlwWXlVVmtmIiwidmVyc2lvbiI6MSwiaWF0IjoxNzU1NzY1ODUwNDA3LCJzdWIiOiJNc3pDSnk0TGZhUlJBbXRXd3l5cCJ9.vPu8roNC4fBhxPL_kEbejgfmR2Cy1qOw92AUrNsW_0c")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID", "sxROcQiT1yHhipYyUVkf")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables. AI features will fail.")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# GHL API Configuration (V2)
GHL_BASE_URL = "https://services.leadconnectorhq.com"
GHL_HEADERS = {
    "Authorization": f"Bearer {GHL_API_KEY}",
    "Version": "2021-07-28",
    "Content-Type": "application/json"
}

# Initialize FastAPI
app = FastAPI(title="Zoom to GHL Integration")

# ------------------------------------------------------------------------------
# DATA MODELS
# ------------------------------------------------------------------------------

class ZoomEventPayload(BaseModel):
    plainToken: Optional[str] = None
    object: Optional[Dict[str, Any]] = None

class ZoomWebhookRequest(BaseModel):
    event: str
    payload: ZoomEventPayload
    download_token: Optional[str] = None

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def get_ghl_contact(email: str) -> Optional[str]:
    """
    Find a GHL Contact ID by email using V2 Search API.
    """
    try:
        url = f"{GHL_BASE_URL}/contacts/"
        params = {
            "locationId": GHL_LOCATION_ID,
            "query": email,
            "limit": 1
        }
        response = requests.get(url, headers=GHL_HEADERS, params=params)
        response.raise_for_status()
        
        data = response.json()
        contacts = data.get("contacts", [])
        
        if contacts:
            return contacts[0]["id"]
        return None
    except Exception as e:
        logger.error(f"Error searching GHL contact for {email}: {e}")
        return None

def create_ghl_note(contact_id: str, note_content: str):
    """
    Post a note to a GHL contact.
    """
    try:
        url = f"{GHL_BASE_URL}/contacts/{contact_id}/notes"
        payload = {
            "body": note_content
        }
        response = requests.post(url, headers=GHL_HEADERS, json=payload)
        response.raise_for_status()
        logger.info(f"Successfully added note to contact {contact_id}")
    except Exception as e:
        logger.error(f"Error creating GHL note: {e}")

def process_recording_logic(download_url: str, email: str):
    """
    Core logic: Download -> Upload to Gemini -> Poll -> Analyze -> Update GHL.
    """
    temp_file_path = None
    file_upload = None

    try:
        logger.info(f"Starting processing for {email}. URL: {download_url}")

        # 1. Download File
        # Use a temporary file to store the video
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp_file_path = tmp.name
            with requests.get(download_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    tmp.write(chunk)
        
        # Check file size
        file_size = os.path.getsize(temp_file_path)
        if file_size < 100:
            logger.error(f"Downloaded file is too small ({file_size} bytes). Aborting.")
            return

        logger.info(f"File downloaded. Size: {file_size} bytes.")

        # 2. Upload to Gemini
        logger.info("Uploading to Gemini...")
        file_upload = genai.upload_file(temp_file_path, mime_type="video/mp4")
        logger.info(f"File uploaded to Gemini: {file_upload.name}")

        # 3. Poll until ACTIVE
        logger.info("Waiting for Gemini file processing...")
        while file_upload.state.name == "PROCESSING":
            time.sleep(5)
            file_upload = genai.get_file(file_upload.name)
        
        if file_upload.state.name != "ACTIVE":
            logger.error(f"Gemini file processing failed. State: {file_upload.state.name}")
            return

        logger.info("Gemini file is ACTIVE. Generating content...")

        # 4. Generate Content (Model: gemini-2.0-flash)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        prompt = (
            "You are an expert business analyst. Analyze this meeting recording.\n"
            "1. Detect the language spoken (Hebrew or English).\n"
            "2. Respond strictly in the detected language.\n"
            "3. Output the response in the following structured format:\n\n"
            "**Language Detected:** [Language]\n\n"
            "**Summary:**\n[Concise Summary]\n\n"
            "**Full Business Plan:**\n[Detailed Actionable Plan]\n\n"
            "**GHL Note:**\n[Short note for CRM context]"
        )

        response = model.generate_content(
            [file_upload, prompt],
            request_options={"timeout": 600}
        )

        result_text = response.text
        logger.info("AI Analysis complete.")

        # 5. Find GHL Contact
        contact_id = get_ghl_contact(email)
        
        if contact_id:
            logger.info(f"Found GHL Contact ID: {contact_id}")
            # 6. Post Note
            create_ghl_note(contact_id, result_text)
        else:
            logger.warning(f"No GHL Contact found for email: {email}. Skipping note creation.")

    except Exception as e:
        logger.error(f"Fatal error in background task: {e}")
    finally:
        # Cleanup local temp file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        # Cleanup Gemini file (optional, but good practice to delete from cloud if strictly transactional)
        if file_upload:
            try:
                genai.delete_file(file_upload.name)
            except Exception:
                pass

# ------------------------------------------------------------------------------
# API ENDPOINTS
# ------------------------------------------------------------------------------

@app.post("/zoom-webhook")
async def zoom_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Handles Zoom Webhooks.
    1. Validates endpoint URL (Handshake).
    2. Processes recording.completed events.
    """
    try:
        body_bytes = await request.body()
        body_json = await request.json()
        
        event = body_json.get("event")
        payload = body_json.get("payload", {})

        # 1. URL Validation (Handshake)
        if event == "endpoint.url_validation":
            plain_token = payload.get("plainToken")
            if not plain_token:
                raise HTTPException(status_code=400, detail="Missing plainToken")
            
            # HMAC SHA-256 hashing
            hash_for_validate = hmac.new(
                ZOOM_WEBHOOK_SECRET.encode("utf-8"),
                plain_token.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            
            return {
                "plainToken": plain_token,
                "encryptedToken": hash_for_validate
            }

        # 2. Recording Completed
        if event == "recording.completed":
            obj = payload.get("object", {})
            
            # Extract Email (Registrant or Host)
            email = obj.get("registrant_email")
            if not email:
                email = obj.get("host_email")
            
            # Extract Download URL (First MP4 file usually)
            recording_files = obj.get("recording_files", [])
            download_url = None
            
            for rf in recording_files:
                # Prefer MP4 video files
                if rf.get("file_type") == "MP4" or rf.get("file_extension") == "MP4":
                    download_url = rf.get("download_url")
                    break
            
            if not download_url and recording_files:
                # Fallback to first available if no MP4 explicitly found
                download_url = recording_files[0].get("download_url")

            if email and download_url:
                # Append access token if Zoom provides it in the download_token field (webhook level)
                # or if it's already in the url.
                # For webhook apps, download_url usually contains a query param token if `recording_files` doesn't enforce OAuth.
                # If "download_token" is in the root payload, we might need to append it `?access_token=...`
                # However, usually Zoom webhook download_url is usable directly or via basic verification.
                # We proceed with the URL provided.
                
                background_tasks.add_task(process_recording_logic, download_url, email)
                logger.info(f"Queued recording processing for {email}")
            else:
                logger.warning("Event received but missing email or download_url.")

            return {"status": "processing_queued"}

        return {"status": "ignored_event"}

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
