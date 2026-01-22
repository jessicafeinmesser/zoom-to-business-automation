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

# ------------------------------------------------------------------------------
# CONFIGURATION & ENVIRONMENT VARIABLES
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment Variables
ZOOM_WEBHOOK_SECRET = os.getenv("ZOOM_WEBHOOK_SECRET", "UR6GqxUNSj-rFvVuQqy9_w")
GHL_API_KEY = os.getenv("GHL_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJsb2NhdGlvbl9pZCI6InN4Uk9jUWlUMXlIaGlwWXlVVmtmIiwidmVyc2lvbiI6MSwiaWF0IjoxNzU1NzY1ODUwNDA3LCJzdWIiOiJNc3pDSnk0TGZhUlJBbXRXd3l5cCJ9.vPu8roNC4fBhxPL_kEbejgfmR2Cy1qOw92AUrNsW_0c")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID", "sxROcQiT1yHhipYyUVkf")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

# GHL API Configuration (V2)
GHL_BASE_URL = "https://services.leadconnectorhq.com"
GHL_HEADERS = {
    "Authorization": f"Bearer {GHL_API_KEY}",
    "Version": "2021-07-28",
    "Content-Type": "application/json"
}

app = FastAPI(title="Zoom to GHL Integration")

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def get_ghl_contact(email: str) -> Optional[str]:
    """Find a GHL Contact ID by email."""
    try:
        url = f"{GHL_BASE_URL}/contacts/"
        params = {"locationId": GHL_LOCATION_ID, "query": email, "limit": 1}
        response = requests.get(url, headers=GHL_HEADERS, params=params)
        response.raise_for_status()
        contacts = response.json().get("contacts", [])
        return contacts[0]["id"] if contacts else None
    except Exception as e:
        logger.error(f"Error searching GHL contact for {email}: {e}")
        return None

def create_ghl_note(contact_id: str, note_content: str):
    """Post a note to a GHL contact."""
    try:
        url = f"{GHL_BASE_URL}/contacts/{contact_id}/notes"
        payload = {"body": note_content}
        response = requests.post(url, headers=GHL_HEADERS, json=payload)
        response.raise_for_status()
        logger.info(f"Successfully added note to contact {contact_id}")
    except Exception as e:
        logger.error(f"Error creating GHL note: {e}")

def process_recording_logic(download_url: str, email: str, download_token: str):
    """Core logic: Download (with token) -> Upload to Gemini -> Analyze -> Update GHL."""
    temp_file_path = None
    file_upload = None

    try:
        logger.info(f"Starting processing for {email}...")

        # 1. Download File with Authorization Token
        # Zoom requires the access_token query param to bypass the login page
        authenticated_url = f"{download_url}?access_token={download_token}"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp_file_path = tmp.name
            with requests.get(authenticated_url, stream=True) as r:
                r.raise_for_status()
                
                # Verify we aren't just downloading a small HTML login page
                content_type = r.headers.get('Content-Type', '')
                if 'text/html' in content_type:
                    logger.error("Download failed: Received HTML instead of a video. Check Zoom Webhook Secret/Token.")
                    return

                for chunk in r.iter_content(chunk_size=8192):
                    tmp.write(chunk)
        
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"File downloaded successfully. Size: {file_size} bytes.")

        if file_size < 100000:  # If less than 100KB, it's likely not a video
            logger.error("Downloaded file is too small to be a recording. Aborting.")
            return

        # 2. Upload to Gemini
        logger.info("Uploading to Gemini...")
        file_upload = genai.upload_file(temp_file_path, mime_type="video/mp4")
        
        # 3. Wait for Gemini processing
        logger.info(f"Waiting for Gemini to process file: {file_upload.name}")
        while file_upload.state.name == "PROCESSING":
            time.sleep(5)
            file_upload = genai.get_file(file_upload.name)
        
        if file_upload.state.name != "ACTIVE":
            logger.error(f"Gemini processing failed. State: {file_upload.state.name}")
            return

        # 4. Generate Content
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

        response = model.generate_content([file_upload, prompt], request_options={"timeout": 600})
        result_text = response.text
        logger.info("AI Analysis complete.")

        # 5. Update GHL
        contact_id = get_ghl_contact(email)
        if contact_id:
            create_ghl_note(contact_id, result_text)
        else:
            logger.warning(f"No GHL Contact found for {email}")

    except Exception as e:
        logger.error(f"Fatal error in background task: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if file_upload:
            try:
                genai.delete_file(file_upload.name)
            except:
                pass

# ------------------------------------------------------------------------------
# API ENDPOINTS
# ------------------------------------------------------------------------------

@app.post("/zoom-webhook")
async def zoom_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        body_json = await request.json()
        event = body_json.get("event")
        payload = body_json.get("payload", {})

        # 1. URL Validation (Handshake)
        if event == "endpoint.url_validation":
            plain_token = payload.get("plainToken")
            hash_for_validate = hmac.new(
                ZOOM_WEBHOOK_SECRET.encode("utf-8"),
                plain_token.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            return {"plainToken": plain_token, "encryptedToken": hash_for_validate}

        # 2. Recording Completed
        if event == "recording.completed":
            # The download_token is at the top level of the payload for recording events
            download_token = body_json.get("download_token")
            obj = payload.get("object", {})
            
            email = obj.get("registrant_email") or obj.get("host_email")
            
            # Find the MP4 file URL
            download_url = None
            recording_files = obj.get("recording_files", [])
            for rf in recording_files:
                if rf.get("file_type") == "MP4" or rf.get("file_extension") == "MP4":
                    download_url = rf.get("download_url")
                    break
            
            if email and download_url and download_token:
                background_tasks.add_task(process_recording_logic, download_url, email, download_token)
                logger.info(f"Queued processing for {email}")
                return {"status": "queued"}
            else:
                logger.warning("Missing required fields (email, url, or token)")

        return {"status": "ignored"}

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
