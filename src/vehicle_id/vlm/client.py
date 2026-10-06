import os
import json
import mimetypes
from dotenv import load_dotenv
import google.generativeai as genai
from google import genai as vertex_genai
from google.genai import types as vertex_types
from vehicle_id.schema import VehicleProfile

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

PROMPT_RAW = """Look at this vehicle image and identify:
- make (manufacturer)
- model
- body_type (sedan, SUV, hatchback, etc.)
- colour

Respond ONLY in this exact JSON format, using null for anything you cannot determine:
{"make": "...", "model": "...", "body_type": "...", "colour": "..."}
"""

PROMPT_CONSTRAINED = """Look at this vehicle image and identify:
- make (manufacturer)
- model
- body type: must be exactly one of sedan, SUV, hatchback, sports, MPV, fastback, estate,
  hardtop convertible, minibus, convertible, pickup, crossover
- colour: must be exactly one of black, silver, white, red, blue, brown, yellow, champagne, green, purple

Respond ONLY in this exact JSON format, using null for anything you cannot determine:
{"make": "...", "model": "...", "body_type": "...", "colour": "..."}
"""

PROMPTS = {"raw": PROMPT_RAW, "constrained": PROMPT_CONSTRAINED}

def predict_profile(image_path: str, model_name: str = MODEL_NAME, prompt: str = "raw"):
    model = genai.GenerativeModel(
        model_name,
        generation_config={"temperature": 0},  # repeatable outputs for evals
    )

    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = model.generate_content([
        PROMPTS[prompt],
        {"mime_type": mime_type, "data": image_bytes},
    ])

    text = response.text
    result = json.loads(text[text.index("{"): text.rindex("}") + 1])

    profile = VehicleProfile(
        make=result.get("make"),
        model=result.get("model"),
        body_type=result.get("body_type"),
        colour=result.get("colour"),
        status={"source": "vlm", "model": model_name}
    )

    usage = response.usage_metadata  # prompt_token_count, candidates_token_count, total_token_count
    return profile, usage


# --- Fine-tuned model (Vertex AI) ------------------------------------------
# Uses the google-genai SDK against Vertex, not the AI-Studio API key above --
# tuned models are only reachable this way. Auth comes from your
# `gcloud auth application-default login` session, not GEMINI_API_KEY.

TUNED_PROJECT = os.getenv("TUNED_PROJECT", "auth-2b5a7")
TUNED_LOCATION = os.getenv("TUNED_LOCATION", "us")  # tuned models serve from the "us" multi-region, not us-central1
TUNED_ENDPOINT = os.getenv(
    "TUNED_ENDPOINT",
    "projects/972693698520/locations/us/endpoints/4884364902115835904",
)

# Must match the prompt used during training (PROMPT_CONSTRAINED) -- the tuned
# model was taught to answer against this exact wording and label list.
PROMPT_TUNED = """Look at this vehicle image and identify:
- make (manufacturer)
- model
- body type: must be exactly one of sedan, SUV, hatchback, sports, MPV, fastback, estate,
  hardtop convertible, minibus, convertible, pickup, crossover

Respond ONLY in this exact JSON format:
{"make": "...", "model": "...", "body_type": "..."}
"""

_tuned_client = None


def _get_tuned_client():
    global _tuned_client
    if _tuned_client is None:
        _tuned_client = vertex_genai.Client(vertexai=True, project=TUNED_PROJECT, location=TUNED_LOCATION)
    return _tuned_client


def predict_profile_tuned(image_path: str, endpoint: str = TUNED_ENDPOINT):
    """Same interface/return shape as predict_profile(), but calls the fine-tuned
    Vertex AI model instead of the zero-shot AI-Studio one. No colour field --
    the tuning set (web-nature manifest) has no colour labels."""
    client = _get_tuned_client()

    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    response = client.models.generate_content(
        model=endpoint,
        contents=[
            vertex_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            PROMPT_TUNED,
        ],
        config=vertex_types.GenerateContentConfig(temperature=0),
    )

    text = response.text
    result = json.loads(text[text.index("{"): text.rindex("}") + 1])

    profile = VehicleProfile(
        make=result.get("make"),
        model=result.get("model"),
        body_type=result.get("body_type"),
        colour=None,
        status={"source": "vlm", "model": "tuned-vehicle-id-flash-lite-v1", "prompt": "tuned"},
    )
    usage = response.usage_metadata
    return profile, usage