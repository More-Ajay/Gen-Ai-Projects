import whisper
import os

WHISPER_MODEL = os.getenv("WHISPER_MODEL" , "small")

_model = None

def load_model():
    
    global _model

    if _model is None:
        print(f"loading model...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("whisper model load successfully")

    return _model

def transcribe_chunk(chunk_path : str , translate : bool = False) -> str:

    model = load_model()

    task = "translate" if translate else "transcribe"

    result = model.transcribe(chunk_path , task = task)

    return result["text"]



def transcribe_all(chunks = list , language="English" , translate : bool = False) -> str:

    full_transcript = ""

    for i , chunk in enumerate(chunks):
        print(f"trancribing chunk {i + 1}")
        
        text = transcribe_chunk(chunk , translate=translate)

        if text is None:
            raise ValueError(
                f"Transcription failed for chunk {i + 1}: "
                "transcribe_chunk returned None."
            )

        full_transcript += text + " "

    print("Transcription Completed")
    return full_transcript.strip()
