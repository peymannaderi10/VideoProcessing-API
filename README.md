# Selfie Video Processing API

A FastAPI application that processes videos using MediaPipe's selfie segmentation model. The service applies different colors to various body parts including hair, skin, clothes, and accessories.

## Features

- **Selfie Segmentation**: Uses the `selfie_multiclass_256x256.tflite` model for multi-class segmentation
- **Video Processing**: Processes entire video files frame by frame
- **Color Coding**: Applies distinct colors to different body parts:
  - Background (black)
  - Hair (blue)
  - Body skin (green)
  - Face skin (red)
  - Clothes (yellow)
  - Accessories (magenta)

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Start the FastAPI server using uvicorn:
```bash
uvicorn main:app --reload
```

The server will start on `http://localhost:8000`

## API Endpoints

### Basic Endpoints
- `GET /` - Returns API information
- `GET /health` - Health check endpoint
- `GET /segmentation/info` - Get segmentation categories and model information

### Video Processing
- `POST /process-video` - Upload and process a video file (saves to downloads folder)
- `GET /download/{filename}` - Download processed video file from downloads folder
- `GET /downloads` - List all processed videos in downloads folder

## Usage

### 1. Check Segmentation Information
```bash
curl http://localhost:8000/segmentation/info
```

### 2. Process a Video
```bash
curl -X POST "http://localhost:8000/process-video" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_video.mp4"
```

**Response:**
```json
{
  "message": "Video processed successfully",
  "original_filename": "your_video.mp4",
  "processed_file": "uuid_processed_your_video.mp4",
  "download_url": "/download/uuid_processed_your_video.mp4",
  "status": "completed"
}
```

### 3. List Processed Videos
```bash
curl http://localhost:8000/downloads
```

### 4. Download Processed Video
```bash
curl -O http://localhost:8000/download/uuid_processed_your_video.mp4
```

**Note:** Processed videos are automatically saved to the `downloads/` folder in the repository and persist between requests.

## Testing with curl

### Basic Tests
```bash
# Health check
curl http://localhost:8000/health

# Get segmentation info
curl http://localhost:8000/segmentation/info

# Simple endpoint
curl http://localhost:8000/
```

### Video Processing Test
```bash
# Process a video file
curl -X POST "http://localhost:8000/process-video" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_video.mp4"
```

## Interactive API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation with Swagger UI.

## Supported Video Formats

- MP4
- AVI
- MOV
- MKV

## Model Information

The service uses the `selfie_multiclass_256x256.tflite` model which provides 6 segmentation categories:
- 0: Background
- 1: Hair
- 2: Body skin
- 3: Face skin
- 4: Clothes
- 5: Accessories

## Requirements

- Python 3.8+
- FastAPI
- MediaPipe
- OpenCV
- NumPy

## File Storage

Processed videos are automatically saved to the `downloads/` folder in the repository. Files are:
- Given unique UUID-based names to avoid conflicts
- Stored permanently (not temporary)
- Accessible via the `/downloads` endpoint or direct download URLs
- Can be manually deleted from the `downloads/` folder if needed

## Troubleshooting

- Ensure video files are in supported formats (MP4, AVI, MOV, MKV)
- Check file sizes (large videos may take time to process)
- Monitor system resources during processing
- Check the `downloads/` folder for processed files
- Use the `/downloads` endpoint to list all available processed videos
