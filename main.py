import os
import json
import hmac
import hashlib
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
import google.generativeai as genai
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, status
from pydantic import BaseModel, Field

# --- Configuration ---
# In a real production environment, these should be environment variables.
# Hardcoded based on specific prompt instructions.
# ZOOM_SECRET = "UR6GqxUNSj-rFvVuQqy9_w"
# GHL_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJsb2NhdGlvbl9pZCI6InN4Uk9jUWlUMXlIaGlwWXlVVmtmIiwidmVyc2lvbiI6MSwiaWF0IjoxNzU1NzY1ODUwNDA3LCJzdWIiOiJNc3pDSnk0TGZhUlJBbXRXd3l5cCJ9.vPu8roNC4fBhxPL_kEbejgfmR2Cy1qOw92AUrNsW_0c"
# GHL_LOCATION_ID = "sxROcQiT1yHhipYyUVkf"
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Assumed to be in env

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Gemini
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    logger.warning("GOOGLE_API_KEY not found in environment variables. processing will fail.")

# Initialize FastAPI
app = FastAPI(title="Zoom to GHL Integration")

# --- Pydantic Models for Zoom Webhook ---

class ZoomPayloadObject(BaseModel):
    plainToken: Optional[str] = None
    registrant_email: Optional[str] = None
    host_email: Optional[str] = None
    topic: Optional[str] = None
    recording_files: Optional[List[Dict[str, Any]]] = None
    download_token: Optional[str] = None

class ZoomPayload(BaseModel):
    object: Optional[ZoomPayloadObject] = None

class ZoomEvent(BaseModel):
    event: str
    payload: ZoomPayload
    event_ts: Optional[int] = None

# --- Helper Functions ---

def validate_zoom_token(plain_token: str, secret: str) -> str:
    """Generates HMAC SHA256 signature for Zoom URL validation."""
    msg = plain_token.encode('utf-8')
    key = secret.encode('utf-8')
    encrypted_token = hmac.new(key, msg, hashlib.sha256).hexdigest()
    return encrypted_token

def download_file(url: str, token: Optional[str] = None) -> str:
    """Downloads file from Zoom to a temporary path."""
    params = {}
    if token:
        params['access_token'] = token
    
    # Note: Zoom download URLs often redirect. Requests handles this automatically.
    with requests.get(url, params=params, stream=True) as r:
        r.raise_for_status()
        # Create a temp file
        fd, path = tempfile.mkstemp(suffix=".mp4")
        with os.fdopen(fd, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
    return path

def process_with_gemini(file_path: str) -> str:
    """Uploads file to Gemini and requests transcription/summary in He/En."""
    if not GOOGLE_API_KEY:
        raise ValueError("Google API Key missing.")

    logger.info(f"Uploading file {file_path} to Gemini...")
    video_file = genai.upload_file(path=file_path)
    
    # Wait for processing (usually fast for audio/video)
    import time
    while video_file.state.name == "PROCESSING":
        time.sleep(2)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError("Gemini file processing failed.")

    logger.info("File processed. Generating content...")
    
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
    prompt = (
        "You are an expert meeting assistant. "
        "1. Transcribe the audio from this meeting. "
        "2. Detect the primary language (Hebrew or English). "
        "3. Provide a concise summary of the meeting in the detected language. "
        "4. Provide the full transcript below the summary. "
        "Format the output clearly with headers: 'Summary' and 'Transcript'."
    )

    response = model.generate_content([video_file, prompt])
    
    # Clean up file from Gemini cloud storage
    genai.delete_file(video_file.name)
    
    return response.text

def ghl_find_contact(email: str) -> Optional[str]:
    """Finds a contact in GHL by email and returns their ID."""
    url = "https://services.leadconnectorhq.com/contacts/search"
    headers = {
        "Authorization": f"Bearer {GHL_API_KEY}",
        "Version": "2021-07-28",
        "Accept": "application/json"
    }
    params = {"q": email} # Search query
    
    try:
        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        
        # contacts array in 'contacts' key
        contacts = data.get('contacts', [])
        if contacts:
            return contacts[0]['id']
        return None
    except Exception as e:
        logger.error(f"Error finding GHL contact: {e}")
        return None

def ghl_create_note(contact_id: str, content: str):
    """Posts the summary/transcript as a note to the GHL contact."""
    url = f"https://services.leadconnectorhq.com/contacts/{contact_id}/notes"
    headers = {
        "Authorization": f"Bearer {GHL_API_KEY}",
        "Version": "2021-07-28",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    payload = {
        "body": content
    }
    
    try:
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        logger.info(f"Successfully posted note to Contact ID {contact_id}")
    except Exception as e:
        logger.error(f"Error creating GHL note: {e}")

def process_recording_task(event_data: ZoomPayloadObject):
    """Background task to handle the recording logic."""
    temp_file_path = None
    try:
        # 1. Identify Email (Participant or Host)
        email = event_data.registrant_email or event_data.host_email
        if not email:
            logger.warning("No email found in payload. Cannot match to GHL.")
            return

        # 2. Identify Download URL (Prefer Audio Only for speed, else MP4)
        files = event_data.recording_files or []
        download_url = None
        for f in files:
            if f.get('file_type') == 'M4A': # Audio only
                download_url = f.get('download_url')
                break
        
        if not download_url and files:
            # Fallback to first available (likely MP4)
            download_url = files[0].get('download_url')
            
        if not download_url:
            logger.warning("No download URL found in recording files.")
            return

        # 3. Download File
        logger.info(f"Downloading recording for {email}...")
        # Zoom usually passes a download_token in the payload if verification is enabled
        temp_file_path = download_file(download_url, event_data.download_token)

        # 4. Process with Gemini (Transcribe + Summarize)
        logger.info("Processing with Gemini...")
        ai_result = process_with_gemini(temp_file_path)

        # 5. GHL Integration
        logger.info(f"Searching GHL for {email}...")
        contact_id = ghl_find_contact(email)
        
        if contact_id:
            logger.info(f"Found Contact ID: {contact_id}. Posting note...")
            ghl_create_note(contact_id, ai_result)
        else:
            logger.warning(f"Contact with email {email} not found in GHL Location {GHL_LOCATION_ID}.")

    except Exception as e:
        logger.error(f"Critical error in background task: {e}", exc_info=True)
    finally:
        # Cleanup local temp file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# --- Routes ---

@app.post("/zoom-webhook")
async def zoom_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # Validate Event Type
    event_type = body.get('event')
    
    # 1. URL Validation
    if event_type == 'endpoint.url_validation':
        plain_token = body.get('payload', {}).get('plainToken')
        if not plain_token:
            raise HTTPException(status_code=400, detail="Missing plainToken")
            
        encrypted_token = validate_zoom_token(plain_token, ZOOM_SECRET)
        return {
            "plainToken": plain_token,
            "encryptedToken": encrypted_token
        }

    # 2. Recording Completed
    elif event_type == 'recording.completed':
        # Parse logic wrapped in try/except to prevent 500 on webhook
        try:
            zoom_event = ZoomEvent(**body)
            background_tasks.add_task(process_recording_task, zoom_event.payload.object)
            return {"status": "processing_started"}
        except Exception as e:
            logger.error(f"Error parsing recording event: {e}")
            # Return 200 to Zoom so they don't retry indefinitely on logic errors
            return {"status": "error_logged"}

    # 3. Other Events
    else:
        # Just acknowledge other events
        return {"status": "ignored"}

if __name__ == "__main__":
    # For debugging purposes
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
