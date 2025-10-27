import whisper
import json

# Load model once globally
_model = None

def get_whisper_model():
    """Get or load the Whisper model (singleton pattern)."""
    global _model
    if _model is None:
        _model = whisper.load_model("medium")
    return _model

def transcribe_video(video_path: str, language: str = "en") -> dict:
    """
    Transcribe a video file using Whisper.
    
    Args:
        video_path: Path to the video file
        language: Language code (default: "en")
    
    Returns:
        Dictionary with transcription text and segments
    """
    model = get_whisper_model()
    
    result = model.transcribe(
        video_path,
        language=language,
        fp16=False
    )
    
    # Simple JSON structure
    transcription_data = {
        "text": result["text"].strip(),
        "segments": [
            {
                "start": segment['start'],
                "end": segment['end'],
                "text": segment['text'].strip()
            }
            for segment in result["segments"]
        ]
    }
    
    return transcription_data