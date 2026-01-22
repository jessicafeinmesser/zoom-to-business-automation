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
# CONFIGURATION & LOGGING
# ------------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment Variables
ZOOM_WEBHOOK_SECRET = os.getenv("ZOOM_WEBHOOK_SECRET", "UR6GqxUNSj-rFvVuQqy9_w")
GHL_API_KEY = os.getenv("GHL_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJsb2NhdGlvbl9pZCI6InN4Uk9jUWlUMXlIaGlwWXlVVmtmIiwidmVyc2lvbiI6MSwiaWF0IjoxNzU1NzY1ODUwNDA3LCJzdWIiOiJNc3pDSnk0TGZhUlJBbXRXd3l5cCJ9.vPu8roNC4fBhxPL_kEbejgfmR2Cy1qOw92AUrNsW_0c")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID", "sxROcQiT1yHhipYyUVkf")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    logger.error("GEMINI_API_KEY not found! Script will fail to analyze recordings.")

genai.configure(api_key=GOOGLE_API_KEY)

# Log available models to help with debugging 404 errors
try:
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    logger.info(f"Available Gemini Models for this API Key: {available_models}")
except Exception as e:
    logger.warning(f"Could not list models: {e}")

# GHL API Configuration (V2)
GHL_BASE_URL = "https://services.leadconnectorhq.com"
GHL_HEADERS = {
    "Authorization": f"Bearer {GHL_API_KEY}",
    "Version": "2021-07-28",
    "Content-Type": "application/json"
}

app = FastAPI(title="Zoom to GHL Integration")

# ------------------------------------------------------------------------------
# GHL HELPERS
# ------------------------------------------------------------------------------

def get_ghl_contact(email: str) -> Optional[str]:
    try:
        url = f"{GHL_BASE_URL}/contacts/"
        params = {"locationId": GHL_LOCATION_ID, "query": email, "limit": 1}
        response = requests.get(url, headers=GHL_HEADERS, params=params)
        response.raise_for_status()
        contacts = response.json().get("contacts", [])
        return contacts[0]["id"] if contacts else None
    except Exception as e:
        logger.error(f"GHL search error for {email}: {e}")
        return None

def create_ghl_note(contact_id: str, note_content: str):
    try:
        url = f"{GHL_BASE_URL}/contacts/{contact_id}/notes"
        payload = {"body": note_content}
        response = requests.post(url, headers=GHL_HEADERS, json=payload)
        response.raise_for_status()
        logger.info(f"GHL Note created for contact {contact_id}")
    except Exception as e:
        logger.error(f"GHL Note creation error: {e}")

# ------------------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------------------

def process_recording_logic(download_url: str, email: str, download_token: str):
    temp_file_path = None
    file_upload = None

    try:
        logger.info(f"Processing recording for: {email}")

        # 1. Download Video
        auth_url = f"{download_url}?access_token={download_token}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            temp_file_path = tmp.name
            with requests.get(auth_url, stream=True) as r:
                r.raise_for_status()
                if 'text/html' in r.headers.get('Content-Type', ''):
                    logger.error("Download failed: Received HTML instead of MP4.")
                    return
                for chunk in r.iter_content(chunk_size=16384):
                    tmp.write(chunk)
        
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"Download complete. Size: {file_size} bytes.")

        # 2. Upload to Gemini
        logger.info("Uploading to Gemini...")
        file_upload = genai.upload_file(temp_file_path, mime_type="video/mp4")
        
        # 3. Poll for Processing
        while file_upload.state.name == "PROCESSING":
            time.sleep(5)
            file_upload = genai.get_file(file_upload.name)
        
        if file_upload.state.name != "ACTIVE":
            logger.error(f"Gemini processing state: {file_upload.state.name}")
            return

        # Settle delay
        time.sleep(10)

        # 4. Generate Analysis
        # Using 'gemini-1.5-flash-latest' to resolve the 404 error
        model_name = "gemini-1.5-flash-latest"
        logger.info(f"Analyzing with {model_name}...")
        model = genai.GenerativeModel(model_name)
        
        prompt = (
            "Analyze this meeting recording and provide:\n"
            "1. Detected Language (Hebrew/English).\n"
            "2. Summary (in the detected language).\n"
            "3. Full Business Plan (in the detected language).\n"
            "4. A short note for a CRM system.\n\n"
            "Maintain the structure and professional tone."
        )

        response = model.generate_content([file_upload, prompt], request_options={"timeout": 600})
        result_text = response.text
        logger.info("AI Analysis completed.")

        # 5. GHL Integration
        contact_id = get_ghl_contact(email)
        if contact_id:
            create_ghl_note(contact_id, result_text)
        else:
            logger.warning(f"No GHL contact found for {email}. Results log: \n{result_text[:200]}...")

    except Exception as e:
        logger.error(f"Background process failed: {e}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if file_upload:
            try:
                genai.delete_file(file_upload.name)
            except:
                pass

# ------------------------------------------------------------------------------
# ENDPOINTS
# ------------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "active", "integration": "Zoom-to-GHL"}

@app.post("/zoom-webhook")
async def zoom_webhook(request: Request, background_tasks: BackgroundTasks):
    try:
        body = await request.json()
        event = body.get("event")
        payload = body.get("payload", {})

        # URL Validation (Handshake)
        if event == "endpoint.url_validation":
            plain_token = payload.get("plainToken")
            hashed = hmac.new(
                ZOOM_WEBHOOK_SECRET.encode("utf-8"),
                plain_token.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            return {"plainToken": plain_token, "encryptedToken": hashed}

        # Recording Logic
        if event == "recording.completed":
            download_token = body.get("download_token")
            obj = payload.get("object", {})
            email = obj.get("registrant_email") or obj.get("host_email")
            
            # Find MP4
            download_url = next((f.get("download_url") for f in obj.get("recording_files", []) 
                                if f.get("file_type") == "MP4"), None)

            if all([email, download_url, download_token]):
                background_tasks.add_task(process_recording_logic, download_url, email, download_token)
                logger.info(f"Queued recording for {email}")
                return {"status": "queued"}
            
        return {"status": "ignored"}

    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
