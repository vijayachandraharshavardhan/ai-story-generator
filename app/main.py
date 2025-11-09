from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from app.story_generator import StoryGenerator
from gtts import gTTS
import os
import uuid
from openai import OpenAI
import asyncio
from dotenv import load_dotenv
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("story_app")

app = FastAPI(redirect_slashes=False)
app.mount("/static", StaticFiles(directory="static"), name="static")

story_generator = StoryGenerator()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SUPPORTED_LANGUAGES = {"en", "es", "fr", "de", "te", "hi"}

class AttentionSpan(str, Enum):
    short = "short"
    medium = "medium"
    long = "long"

class ReadingLevel(str, Enum):
    basic = "basic"
    intermediate = "intermediate"
    advanced = "advanced"

class StoryRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=30)
    age: int = Field(..., ge=1, le=12)
    genre: str = Field(..., min_length=3, max_length=30)
    language: str = Field("en")
    prompt: Optional[str] = Field(None, max_length=500)
    attention_span: AttentionSpan = AttentionSpan.medium
    reading_level: ReadingLevel = ReadingLevel.basic

@app.post("/generate_story_audio")
async def generate_story_audio(request: StoryRequest, background_tasks: BackgroundTasks):
    try:
        if request.language not in SUPPORTED_LANGUAGES:
            raise HTTPException(status_code=400, detail=f"Unsupported language: {request.language}")

        story = story_generator.generate_story(
            name=request.name,
            age=request.age,
            genre=request.genre,
            language=request.language,
            custom_prompt=request.prompt,
            attention_span=request.attention_span.value,
            reading_level=request.reading_level.value
        )

        filename = f"story_{uuid.uuid4().hex}.mp3"
        filepath = os.path.join("static", filename)

        # Generate audio in background thread to keep response fast
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: gTTS(text=story, lang=request.language).save(filepath))

        image_url = ""
        try:
            response = openai_client.images.generate(
                prompt=f"Children's storybook illustration: {story[:300]}",
                n=1,
                size="512x512"
            )
            image_url = response.data[0].url
        except Exception as e:
            logger.error(f"Image generation error: {e}")

        return {
            "story": story,
            "audio_url": f"/static/{filename}",
            "image_url": image_url
        }

    except Exception as e:
        logger.error(f"Error generating story/audio/image: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def home():
    return {"message": "Welcome to AI Story Teller! Access frontend at /static/index.html"}
