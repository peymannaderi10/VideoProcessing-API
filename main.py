from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os
import tempfile
import shutil
import uuid
import asyncio
from typing import Optional
from video_processing import get_processor, process_video_file
from transcription import transcribe_video

app = FastAPI(title="Selfie Video Processing API", version="1.0.0")

# Create downloads directory if it doesn't exist
DOWNLOADS_DIR = os.path.join(os.path.dirname(__file__), "downloads")
os.makedirs(DOWNLOADS_DIR, exist_ok=True)

@app.get("/")
async def hello_world():
    return {"message": "Selfie Video Processing API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/segmentation/info")
async def get_segmentation_info():
    """Get information about the segmentation categories and model."""
    try:
        processor = get_processor()
        return processor.get_segmentation_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting segmentation info: {str(e)}")

@app.post("/mask-video")
async def mask_video_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload and mask a video file with selfie segmentation.

    The video will be processed with the selfie segmentation model and colors will be applied
    to different body parts (hair, skin, clothes, etc.).

    Returns the path to the masked video file in the downloads folder.
    """
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a video file (mp4, avi, mov, mkv)")

    # Create temporary input file
    temp_input = None
    processed_filename = None

    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_input = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        # Generate unique filename for processed video
        file_id = str(uuid.uuid4())
        processed_filename = f"{file_id}_processed_{os.path.splitext(file.filename)[0]}.mp4"
        processed_path = os.path.join(DOWNLOADS_DIR, processed_filename)

        # Process the video
        success, output_path = process_video_file(temp_input, processed_path)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to process video")

        # Clean up temporary input file
        if temp_input and os.path.exists(temp_input):
            os.unlink(temp_input)

        return {
            "message": "Video processed successfully",
            "original_filename": file.filename,
            "processed_file": processed_filename,
            "download_url": f"/download/{processed_filename}",
            "status": "completed"
        }

    except Exception as e:
        # Cleanup on error
        if temp_input and os.path.exists(temp_input):
            os.unlink(temp_input)
        if processed_filename and os.path.exists(os.path.join(DOWNLOADS_DIR, processed_filename)):
            os.unlink(os.path.join(DOWNLOADS_DIR, processed_filename))
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/process-video")
async def process_video_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = "en"
):
    """
    Upload and process a video file with both segmentation masking and transcription.

    The video will be:
    1. Transcribed using the Whisper model
    2. Processed with the selfie segmentation model (masking)

    Returns both the masked video file path and the transcription.
    """
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a video file (mp4, avi, mov, mkv)")

    # Create temporary input file
    temp_input = None
    processed_filename = None

    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_input = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        # Generate unique filename for processed video
        file_id = str(uuid.uuid4())
        processed_filename = f"{file_id}_processed_{os.path.splitext(file.filename)[0]}.mp4"
        processed_path = os.path.join(DOWNLOADS_DIR, processed_filename)

        # Run transcription and video masking in parallel
        async def run_transcription():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, transcribe_video, temp_input, language)

        async def run_masking():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, process_video_file, temp_input, processed_path)

        # Execute both tasks concurrently
        transcription_result, (success, output_path) = await asyncio.gather(
            run_transcription(),
            run_masking()
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to process video")

        # Clean up temporary input file
        if temp_input and os.path.exists(temp_input):
            os.unlink(temp_input)

        return {
            "message": "Video processed and transcribed successfully",
            "original_filename": file.filename,
            "masked_video": {
                "filename": processed_filename,
                "download_url": f"/download/{processed_filename}"
            },
            "transcription": transcription_result,
            "status": "completed"
        }

    except Exception as e:
        # Cleanup on error
        if temp_input and os.path.exists(temp_input):
            os.unlink(temp_input)
        if processed_filename and os.path.exists(os.path.join(DOWNLOADS_DIR, processed_filename)):
            os.unlink(os.path.join(DOWNLOADS_DIR, processed_filename))
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/transcribe-video")
async def transcribe_video_endpoint(
    file: UploadFile = File(...),
    language: str = "en"
):
    """
    Upload and transcribe a video file.

    The video will be transcribed using the Whisper model.

    Returns the transcription text and segments with timestamps.
    """
    # Validate file type
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.mp3', '.wav', '.m4a')):
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a video or audio file")

    # Create temporary input file
    temp_input = None

    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_input = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        # Transcribe the video
        transcription_result = transcribe_video(temp_input, language=language)

        # Clean up temporary input file
        if temp_input and os.path.exists(temp_input):
            os.unlink(temp_input)

        return {
            "message": "Video transcribed successfully",
            "original_filename": file.filename,
            "transcription": transcription_result,
            "status": "completed"
        }

    except Exception as e:
        # Cleanup on error
        if temp_input and os.path.exists(temp_input):
            os.unlink(temp_input)
        raise HTTPException(status_code=500, detail=f"Error transcribing video: {str(e)}")

@app.get("/download/{filename}")
async def download_processed_video(filename: str):
    """Download a processed video file from the downloads folder."""
    file_path = os.path.join(DOWNLOADS_DIR, filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='video/mp4'
    )

@app.get("/downloads")
async def list_downloads():
    """List all processed videos in the downloads folder."""
    try:
        files = []
        for filename in os.listdir(DOWNLOADS_DIR):
            file_path = os.path.join(DOWNLOADS_DIR, filename)
            if os.path.isfile(file_path):
                files.append({
                    "filename": filename,
                    "size": os.path.getsize(file_path),
                    "download_url": f"/download/{filename}"
                })
        return {"files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing downloads: {str(e)}")

# Processed videos are now stored permanently in the downloads folder
# No cleanup needed for processed files
