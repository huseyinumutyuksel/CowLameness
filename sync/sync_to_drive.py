"""
Google Drive Synchronization Script
Uploads ONLY CSV/H5 outputs to Drive (Satır 24 - videos are already there)

Uploads:
    DeepLabCut/outputs/*.csv → Drive/outputs/deeplabcut/
    MMPose/outputs/*.csv → Drive/outputs/mmpose/
"""
import os
import sys
from pathlib import Path
from typing import List
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from google.oauth2.credentials import Credentials
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from google.auth.transport.requests import Request
    import pickle
except ImportError:
    logger.error("❌ Google Drive API not installed")
    logger.error("   Install with: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
    sys.exit(1)

# Configuration
DRIVE_FOLDER_NAME = "Inek Topallik Tespiti Parcalanmis Inek Videolari"
LOCAL_DLC_OUTPUT = Path("../DeepLabCut/outputs")
LOCAL_MMPOSE_OUTPUT = Path("../MMPose/outputs")

# Scopes
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_credentials():
    """Get or refresh Google Drive credentials"""
    creds = None
    token_path = Path("token.pickle")
    
    # Token file stores user's access and refresh tokens
    if token_path.exists():
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials, let user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            from google_auth_oauthlib.flow import InstalledAppFlow
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for next run
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
    
    return creds

def find_or_create_folder(service, folder_name: str, parent_id: str = None):
    """Find existing folder or create new one"""
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name)'
    ).execute()
    
    items = results.get('files', [])
    
    if items:
        logger.info(f"✅ Found existing folder: {folder_name}")
        return items[0]['id']
    else:
        # Create folder
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            file_metadata['parents'] = [parent_id]
        
        folder = service.files().create(
            body=file_metadata,
            fields='id'
        ).execute()
        
        logger.info(f"✅ Created folder: {folder_name}")
        return folder.get('id')

def upload_file(service, file_path: Path, folder_id: str):
    """Upload a single file to Drive folder"""
    file_metadata = {
        'name': file_path.name,
        'parents': [folder_id]
    }
    
    media = MediaFileUpload(
        str(file_path),
        mimetype='text/csv' if file_path.suffix == '.csv' else 'application/octet-stream',
        resumable=True
    )
    
    try:
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        logger.info(f"  ✅ Uploaded: {file_path.name}")
        return file.get('id')
    except Exception as e:
        logger.error(f"  ❌ Failed to upload {file_path.name}: {e}")
        return None

def sync_outputs():
    """Main sync function"""
    print("\n" + "="*60)
    print("🔄 GOOGLE DRIVE SYNC")
    print("="*60)
    
    # Get credentials
    logger.info("🔐 Authenticating with Google Drive...")
    try:
        creds = get_credentials()
        service = build('drive', 'v3', credentials=creds)
        logger.info("✅ Authentication successful")
    except Exception as e:
        logger.error(f"❌ Authentication failed: {e}")
        logger.error("\nPlease ensure:")
        logger.error("  1. You have 'credentials.json' file in this directory")
        logger.error("  2. Follow setup instructions in README.md")
        sys.exit(1)
    
    # Find base folder
    logger.info(f"\n📁 Finding base folder: {DRIVE_FOLDER_NAME}")
    base_folder_id = find_or_create_folder(service, DRIVE_FOLDER_NAME)
    
    # Find/create outputs folder
    outputs_folder_id = find_or_create_folder(service, "outputs", base_folder_id)
    
    # Find/create deeplabcut and mmpose folders
    dlc_folder_id = find_or_create_folder(service, "deeplabcut", outputs_folder_id)
    mmpose_folder_id = find_or_create_folder(service, "mmpose", outputs_folder_id)
    
    # Upload DeepLabCut outputs
    logger.info("\n📤 Uploading DeepLabCut outputs...")
    dlc_files = list(LOCAL_DLC_OUTPUT.glob("*.csv"))
    logger.info(f"   Found {len(dlc_files)} CSV files")
    
    dlc_uploaded = 0
    for csv_file in dlc_files:
        if upload_file(service, csv_file, dlc_folder_id):
            dlc_uploaded += 1
    
    # Upload MMPose outputs
    logger.info("\n📤 Uploading MMPose outputs...")
    mmpose_files = list(LOCAL_MMPOSE_OUTPUT.glob("*.csv"))
    logger.info(f"   Found {len(mmpose_files)} CSV files")
    
    mmpose_uploaded = 0
    for csv_file in mmpose_files:
        if upload_file(service, csv_file, mmpose_folder_id):
            mmpose_uploaded += 1
    
    # Summary
    print("\n" + "="*60)
    print("✅ SYNC COMPLETE")
    print("="*60)
    print(f"DeepLabCut: {dlc_uploaded}/{len(dlc_files)} files uploaded")
    print(f"MMPose: {mmpose_uploaded}/{len(mmpose_files)} files uploaded")
    print("\nNext step: Run Colab notebook for training")
    print("  Open: Colab_Notebook/Cow_Lameness_Analysis_v18.ipynb")
    print("="*60)

if __name__ == "__main__":
    if not Path("credentials.json").exists():
        print("\n" + "="*60)
        print("❌ ERROR: credentials.json not found")
        print("="*60)
        print("\nTo setup Google Drive API:")
        print("1. Go to: https://console.cloud.google.com/")
        print("2. Create a project")
        print("3. Enable Google Drive API")
        print("4. Create OAuth 2.0 credentials")
        print("5. Download as 'credentials.json'")
        print("6. Place in this directory (sync/)")
        print("\nSee README.md for detailed instructions")
        print("="*60)
        sys.exit(1)
    
    sync_outputs()
