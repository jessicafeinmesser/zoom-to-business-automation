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
ZOOM_WEBHOOK_SECRET = os.getenv("ZOOM_WEBHOOK_SECRET", "UR6GqxUNSj-rFvVuQqy9_w")
GHL_API_KEY = os.getenv("GHL_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJsb2NhdGlvbl9pZCI6InN4Uk9jUWlUMXlIaGlwWXlVVmtmIiwidmVyc2lvbiI6MSwiaWF0IjoxNzU1NzY1ODUwNDA3LCJzdWIiOiJNc3pDSnk0TGZhUlJBbXRXd3l5cCJ9.vPu8roNC4fBhxPL_kEbejgfmR2Cy1qOw92AUrNsW_0c")
GHL_LOCATION_ID = os.getenv("GHL_LOCATION_ID", "sxROcQiT1yHhipYyUVkf")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    logger.error("GEMINI_API_KEY missing!")

genai.configure(api_key=GOOGLE_API_KEY)

# GHL API Configuration
GHL_BASE_URL = "https://services.leadconnectorhq.com"
GHL_HEADERS = {
    "Authorization": f"Bearer {GHL_API_KEY}",
    "Version": "2021-07-28",
    "Content-Type": "application/json"
}

app = FastAPI()

# ------------------------------------------------------------------------------
# HELPERS
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
        logger.error(f"GHL Search Error: {e}")
        return None

def create_ghl_note(contact_id: str, note_content: str):
    try:
        url = f"{GHL_BASE_URL}/contacts/{contact_id}/notes"
        payload = {"body": note_content}
        response = requests.post(url, headers=GHL_HEADERS, json=payload)
        response.raise_for_status()
        logger.info(f"GHL Note added to {contact_id}")
    except Exception as e:
        logger.error(f"GHL Note Error: {e}")

# ------------------------------------------------------------------------------
# CORE LOGIC
# ------------------------------------------------------------------------------

def process_recording_logic(download_url: str, email: str, download_token: str):
    temp_file_path = None
    file_upload = None

    try:
        logger.info(f"Processing recording for {email}")

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
        logger.info("Uploading to Gemini...")
        file_upload = genai.upload_file(temp_file_path, mime_type="video/mp4")
        
        while file_upload.state.name == "PROCESSING":
            time.sleep(5)
            file_upload = genai.get_file(file_upload.name)
        
        if file_upload.state.name != "ACTIVE":
            logger.error(f"File failed to become active: {file_upload.state.name}")
            return

        time.sleep(10) # Settle delay

        # 3. Dynamic Model Selection (FIXED FOR YOUR KEY)
        available_names = [m.name for m in genai.list_models()]
        
        # We look for 'gemini-flash-latest' first as it was in your log list
        if "models/gemini-flash-latest" in available_names:
            chosen_model = "models/gemini-flash-latest"
        elif "models/gemini-1.5-flash" in available_names:
            chosen_model = "models/gemini-1.5-flash"
        else:
            # Fallback: pick the first thing that looks like a flash model
            chosen_model = next((name for name in available_names if "flash" in name), "models/gemini-1.5-flash")

        # 4. Generate Content
        logger.info(f"Requesting generation from {chosen_model}...")
        model = genai.GenerativeModel(model_name=chosen_model)
        
        prompt = (
            "Analyze this meeting recording and provide:\n"
            "1. Detected Language (Hebrew/English).\n"
            "2. Summary (in the detected language).\n"
            "3. Full Business Plan (in the detected language).\n"
            "4. A short note for a CRM system.\n\n"
            "Maintain the structure and professional tone."
        )

        response = model.generate_content([file_upload, prompt], request_options={"timeout": 600})
        
        if not response.text:
            logger.error("AI returned empty response.")
            return

        # 5. Send to GHL
        contact_id = get_ghl_contact(email)
        if contact_id:
            create_ghl_note(contact_id, response.text)
        else:
            logger.warning(f"Analysis generated but no GHL contact found for {email}.")

    except Exception as e:
        logger.error(f"Process Error: {e}")
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
            email = obj.get("registrant_email") or obj.get("host_email")
            files = obj.get("recording_files", [])
            download_url = next((f.get("download_url") for f in files if f.get("file_type") == "MP4"), None)

            if email and download_url and download_token:
                background_tasks.add_task(process_recording_logic, download_url, email, download_token)
                logger.info(f"Webhook received. Queued processing for {email}")
                return {"status": "queued"}

        return {"status": "ignored"}
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return {"error": str(e)}

@app.get("/")
def home():
    return {"status": "online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
