import os
import hmac
import hashlib
import json
import time
import logging
import requests
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, BackgroundTasks, HTTPException, Header
from pydantic import BaseModel
import google.generativeai as genai

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment Variables
ZOOM_WEBHOOK_SECRET = os.getenv("ZOOM_WEBHOOK_SECRET", "UR6GqxUNSj-rFvVuQqy9_w")
GHL_API_KEY = os.getenv("GHL_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJsb2NhdGlvbl9pZCI6InN4Uk9jUWlUMXlIaGlwWXlVVmtmIiwidmVyc2lvbiI6MSwiaWF0IjoxNzU1NzY1ODUwNDA3LCJzdWIiOiJNc3pDSnk0TGZhUlJBbXRXd3l5cCJ9.vPu8roNC4fBhxPL_kEbejgfmR2Cy1qOw92AUrNsW_0c")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID", "sxROcQiT1yHhipYyUVkf")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
if not GOOGLE_API_KEY:
    logger.error("GEMINI_API_KEY is not set.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Initialize FastAPI
app = FastAPI(title="Zoom to GHL Integration")

# Pydantic Models for Zoom Webhook
class ZoomPayloadObject(BaseModel):
    plainToken: Optional[str] = None
    id: Optional[int] = None
    uuid: Optional[str] = None
    host_email: Optional[str] = None
    topic: Optional[str] = None
    start_time: Optional[str] = None
    duration: Optional[int] = None
    share_url: Optional[str] = None
    recording_files: Optional[List[Dict[str, Any]]] = None
    download_token: Optional[str] = None
    registrant_email: Optional[str] = None 

class ZoomPayload(BaseModel):
    account_id: Optional[str] = None
    object: Optional[ZoomPayloadObject] = None

class ZoomWebhookEvent(BaseModel):
    event: str
    payload: ZoomPayload
    download_token: Optional[str] = None

# --- Core Logic Functions ---

def get_ghl_headers():
    return {
        "Authorization": f"Bearer {GHL_API_KEY}",
        "Version": "2021-07-28",
        "Content-Type": "application/json"
    }

def find_ghl_contact(email: str) -> Optional[str]:
    """Finds a contact in GHL by email using V2 API. Returns Contact ID."""
    url = "https://services.leadconnectorhq.com/contacts/search"
    # Note: GHL V2 Search usually requires a POST or specific query params. 
    # Using the standard V2 'search' endpoint which accepts query params for email.
    params = {
        "email": email,
        "locationId": GHL_LOCATION_ID
    }
    
    try:
        response = requests.get(url, headers=get_ghl_headers(), params=params)
        response.raise_for_status()
        data = response.json()
        contacts = data.get("contacts", [])
        
        if contacts:
            return contacts[0]["id"]
        return None
    except Exception as e:
        logger.error(f"Error searching GHL contact: {e}")
        return None

def create_ghl_note(contact_id: str, note_content: str):
    """Creates a note for a specific contact in GHL."""
    url = f"https://services.leadconnectorhq.com/contacts/{contact_id}/notes"
    payload = {
        "body": note_content,
        "userId": "" # Optional: Assign to a specific user if needed
    }
    
    try:
        response = requests.post(url, headers=get_ghl_headers(), json=payload)
        response.raise_for_status()
        logger.info(f"Note successfully posted to GHL Contact ID: {contact_id}")
    except Exception as e:
        logger.error(f"Error posting GHL note: {e}")

def process_recording_background(payload_obj: ZoomPayloadObject, download_token: str):
    """
    Background task to download video, process with Gemini, and update GHL.
    """
    temp_file_path = None
    uploaded_file = None

    try:
        # 1. Identify valid recording file (prefer MP4)
        if not payload_obj.recording_files:
            logger.warning("No recording files found in payload.")
            return

        # Find the largest MP4 file or the first recording file
        video_info = next((f for f in payload_obj.recording_files if f.get('file_type') == 'MP4'), payload_obj.recording_files[0])
        download_url = video_info.get('download_url')
        
        if not download_url:
            logger.error("No download URL found.")
            return

        # Append access token if provided (Zoom often requires this for webhook downloads)
        final_download_url = f"{download_url}?access_token={download_token}" if download_token else download_url

        # 2. Download File
        logger.info(f"Starting download from {download_url}...")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            temp_file_path = tmp_file.name
            with requests.get(final_download_url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
        
        # 3. Check File Size (< 100 bytes check)
        file_size = os.path.getsize(temp_file_path)
        if file_size < 100:
            logger.error(f"Downloaded file is too small ({file_size} bytes). Likely an error page or empty recording. Stopping.")
            return
        
        logger.info(f"Download complete. Size: {file_size} bytes.")

        # 4. Upload to Gemini
        logger.info("Uploading to Gemini...")
        uploaded_file = genai.upload_file(temp_file_path, mime_type="video/mp4")
        logger.info(f"Uploaded to Gemini. URI: {uploaded_file.uri}")

        # 5. Poll until ACTIVE
        while True:
            uploaded_file = genai.get_file(uploaded_file.name)
            if uploaded_file.state.name == "ACTIVE":
                break
            if uploaded_file.state.name == "FAILED":
                logger.error("Gemini file processing FAILED.")
                return
            logger.info("Waiting for Gemini video processing...")
            time.sleep(10)

        # 6. Generate Content
        logger.info("Generating content with Gemini 1.5 Flash...")
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = (
            "Analyze this meeting recording. "
            "1. Detect the language spoken (Hebrew or English). "
            "2. Generate a response in the DETECTED LANGUAGE. "
            "3. Provide the output in the following format:\n\n"
            "---SUMMARY---\n[Detailed Summary]\n\n"
            "---BUSINESS_PLAN---\n[Full Business Plan based on discussion]\n\n"
            "---SHORT_NOTE---\n[A concise note for the CRM]"
        )

        result = model.generate_content([uploaded_file, prompt])
        ai_response = result.text

        # 7. Find Contact and Post Note
        # Logic: Try registrant_email first (if available), then host_email
        target_email = payload_obj.registrant_email or payload_obj.host_email
        
        if not target_email:
            logger.warning("No email found in payload to associate with GHL.")
            return

        logger.info(f"Searching GHL for contact: {target_email}")
        contact_id = find_ghl_contact(target_email)

        if contact_id:
            logger.info(f"Contact found ({contact_id}). Posting note...")
            
            # Formating the note for GHL
            formatted_note = (
                f"Zoom Meeting Analysis\n"
                f"Topic: {payload_obj.topic}\n"
                f"Date: {payload_obj.start_time}\n\n"
                f"{ai_response}"
            )
            create_ghl_note(contact_id, formatted_note)
        else:
            logger.warning(f"Contact with email {target_email} not found in GHL.")

    except Exception as e:
        logger.exception(f"Error in background processing: {e}")
    finally:
        # Cleanup local file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        # Cleanup Gemini file (Optional but recommended to save storage limits)
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
            except Exception:
                pass

# --- Webhook Endpoint ---

@app.post("/zoom-webhook")
async def zoom_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_zm_signature: Optional[str] = Header(None),
    x_zm_request_timestamp: Optional[str] = Header(None)
):
    try:
        body_bytes = await request.body()
        body_json = await request.json()
        
        # 1. URL Validation (Zoom Challenge)
        if body_json.get("event") == "endpoint.url_validation":
            plain_token = body_json["payload"]["plainToken"]
            
            # Construct message for HMAC
            msg = plain_token
            digest = hmac.new(
                ZOOM_WEBHOOK_SECRET.encode("utf-8"),
                msg.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            
            return {
                "plainToken": plain_token,
                "encryptedToken": digest
            }

        # 2. Verify Request Signature for other events (Security Best Practice)
        # Note: Zoom signature verification logic involves `v0:{timestamp}:{body}`
        if x_zm_signature and x_zm_request_timestamp:
            message = f"v0:{x_zm_request_timestamp}:{body_bytes.decode('utf-8')}"
            hashed = hmac.new(
                ZOOM_WEBHOOK_SECRET.encode('utf-8'), 
                message.encode('utf-8'), 
                hashlib.sha256
            ).hexdigest()
            signature = f"v0={hashed}"
            
            if signature != x_zm_signature:
                logger.warning("Invalid Zoom Signature")
                # Depending on strictness, might want to raise HTTPException(401)
                # For now, we proceed or log.

        # 3. Handle Recording Completed
        if body_json.get("event") == "recording.completed":
            # Parse payload safely
            event_data = ZoomWebhookEvent(**body_json)
            payload_obj = event_data.payload.object
            
            # Get download token either from top level or object
            # Note: Webhook structure varies slightly by Zoom app settings, check both
            d_token = event_data.download_token or payload_obj.download_token

            if payload_obj:
                background_tasks.add_task(process_recording_background, payload_obj, d_token)
            
            return {"status": "processing_started"}

        return {"status": "event_received"}

    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
