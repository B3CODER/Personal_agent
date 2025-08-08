from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from dotenv import load_dotenv
import os

from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_config
from langmem import create_memory_store_manager
from sentence_transformers import SentenceTransformer

# ---- Memory Manager ----
from memory_manager import SummaryManager
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Summary prompt
summary_prompt = PromptTemplate.from_template("""
You are an AI assistant tasked with summarizing a multi-turn conversation.

Please generate a **factually accurate and concise summary** that:
- Preserves important facts, names, figures, decisions, and goals
- Captures user intentions and steps taken
- Avoids hallucinations or adding information not explicitly stated
                                              
When creating the summary do not write initial messages like "This is your summary" or "Based on the previous conversation"
- Just generate the summary without the initial messages.
- Only include the summary in the response and not any additional messages.

Existing Summary:
{summary}

New Dialogue to Incorporate:
{new_lines}

Updated Summary:
""")

# ----------------- User Profile Model -----------------
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime

class TravelProfile(BaseModel):
    # ===== Basic Preferences =====
    travel_style: Optional[str] = None  # "budget", "luxury", etc.
    accommodation_type: Optional[List[str]] = None  # ["hotels", "airbnb"]
    group_size_preference: Optional[str] = None  # "solo", "couple", etc.
    travel_pace: Optional[str] = None  # "fast", "moderate", "slow"
    planning_style: Optional[str] = None  # "detailed", "spontaneous", etc.

    # ===== Travel History =====
    visited_destinations: Optional[List[Dict[str, str]]] = None  # [{"country": "Japan", "city": "Tokyo", "date": "2023-05"}]
    favorite_destinations: Optional[List[str]] = None
    disappointing_destinations: Optional[List[str]] = None
    revisit_list: Optional[List[str]] = None
    return_to_destinations: Optional[List[str]] = None
    trip_frequency: Optional[str] = None  # "monthly", etc.
    typical_trip_duration: Optional[str] = None  # "weekend", "2_weeks"
    typical_vacation_length_days: Optional[int] = None
    medical_conditions_to_account: Optional[List[str]] = None  # ["asthma", "low altitude only"]
    needs_digital_connectivity: Optional[bool] = None  # "Yes" if they need Wi-Fi or remote work tools


    # ===== Interests and Activities =====
    preferred_activities: Optional[List[str]] = None  # ["hiking", "beaches"]
    cultural_interests: Optional[List[str]] = None  # ["art", "festivals"]
    special_interests: Optional[List[str]] = None  # ["photography", "diving"]
    activity_level: Optional[str] = None  # "moderate", "active", etc.

    # ===== Practical & Budget Constraints =====
    budget_range: Optional[str] = None  # "moderate", "luxury", etc.
    budget_per_day_usd: Optional[int] = None
    spending_categories: Optional[Dict[str, float]] = None  # {"food": 100, "shopping": 200}
    dietary_restrictions: Optional[List[str]] = None  # ["vegetarian", "halal"]
    mobility_constraints: Optional[str] = None  # "moderate", "significant"

    # ===== Transportation Preferences =====
    preferred_transport_modes: Optional[List[str]] = None  # ["train", "plane"]
    flight_preferences: Optional[Dict[str, str]] = None  # {"class": "economy", "type": "direct"}
    preferred_airlines: Optional[List[str]] = None
    local_transport_options: Optional[List[str]] = None  # ["rental_car", "tours"]
    max_flight_duration_hours: Optional[int] = None  # e.g. 8

    # ===== Environment & Climate =====
    climate_preferences: Optional[List[str]] = None  # ["cold", "tropical"]
    preferred_seasons: Optional[List[str]] = None  # ["summer", "spring"]
    geography_preferences: Optional[List[str]] = None  # ["beach", "mountain"]
    crowd_tolerance: Optional[str] = None  # "prefers_quiet", "loves_crowds", etc.

    # ===== Social & Cultural Preferences =====
    cultural_adventurousness: Optional[str] = None  # "moderate", "familiar_only"
    food_adventurousness: Optional[str] = None  # "very_adventurous", etc.
    language_comfort_level: Optional[str] = None  # "only_english", "multilingual"
    social_preference: Optional[str] = None  # "prefers_privacy", etc.

    # ===== Trip Intent / Purpose =====
    typical_travel_companions: Optional[List[str]] = None  # ["spouse", "family"]
    trip_purposes: Optional[List[str]] = None  # ["relaxation", "business"]
    special_occasions: Optional[List[str]] = None  # ["birthday", "honeymoon"]

    # ===== Booking Behavior =====
    booking_advance_notice: Optional[str] = None  # "last_minute", "months"
    price_sensitivity_level: Optional[str] = None  # "very_sensitive", etc.
    preferred_booking_platforms: Optional[List[str]] = None  # ["booking.com", "direct"]

    # ===== Personality & Behavior Traits =====
    travel_experience_level: Optional[str] = None  # "beginner", "expert"
    revisits_preference: Optional[str] = None  # "loves_new_places", etc.
    trip_spontaneity_level: Optional[str] = None  # "structured", "spontaneous"
    daily_rhythm: Optional[str] = None  # "early_riser", "night_owl", "neutral"

    # ===== Destination Intentions =====
    bucket_list_destinations: Optional[List[str]] = None
    avoid_regions: Optional[List[str]] = None
    preferred_continents: Optional[List[str]] = None

    # ===== Availability / Calendar =====
    travel_blackout_dates: Optional[List[str]] = None  # ["2025-01-01"]
    prefers_long_weekends: Optional[bool] = None
    available_holidays_per_year: Optional[int] = None

    # ===== Personalization Behavior =====
    exploration_vs_familiarity: Optional[str] = None  # "explore", "repeat_favorites"
    surprise_trip_tolerance: Optional[str] = None  # "loves_surprises", "needs_plan"
    language_assistance_required: Optional[bool] = None
    trip_customization_preference: Optional[str] = None  # "custom", "prebuilt"

    # ===== Sharing & Collaboration =====
    shareable_profile_enabled: Optional[bool] = None
    collaborative_trip_planning_allowed: Optional[bool] = None

    # ===== AI Insights / Enhancements (Optional) =====
    inferred_travel_persona: Optional[str] = None  # e.g., "Explorer", "Relaxer"
    ai_generated_profile_summary: Optional[str] = None
    trip_feedback_history: Optional[List[Dict[str, str]]] = None  # [{"trip": "Paris", "rating": "positive"}]

    # ===== Legal & Geo Data =====
    home_country: Optional[str] = None
    passport_country: Optional[str] = None
    visa_free_countries: Optional[List[str]] = None
    visa_restricted_countries: Optional[List[str]] = None
    disliked_regions: Optional[List[str]] = None

    # ===== System / Metadata =====
    profile_last_updated: Optional[datetime] = None
    profile_completeness_score: Optional[float] = None  # e.g. 0.85 meaning 85% filled
# ----------------- Embeddings Store -----------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def embed_fn(texts):
    return embed_model.encode(texts, convert_to_numpy=True).tolist()

store = InMemoryStore(
    index={"dims": 384, "embed": embed_fn}
)

manager = create_memory_store_manager(
    "google_genai:gemini-2.5-flash",
    namespace=("users", "{user_id}", "profile"),
    schemas=[TravelProfile],
    instructions="Extract all user information and events as triples.",
    enable_inserts=True,
    enable_deletes=True,
)

my_llm = init_chat_model("google_genai:gemini-2.5-flash")

# Initialize summary manager with hierarchical logic
summary_manager = SummaryManager(my_llm, summary_prompt, max_chunks=5)

@entrypoint(store=store)
def chat(messages: list):
    configurable = get_config()["configurable"]

    # Load user profile
    results = store.search(("users", configurable["user_id"], "profile"))
    profile = None
    if results:
        profile = f"<User Profile>:\n{results[0].value}\n</User Profile>"

    # Load conversation summaries
    conversation_context = summary_manager.get_full_summary()
    if conversation_context.strip():
        conversation_context = f"<Conversation Context>:\n{conversation_context}\n</Conversation Context>"

    # Build system prompt
    system_prompt = (
        "You are a helpful travel assistant.\n"
        "Do NOT return structured profile JSON to the user — only respond in natural language unless explicitly asked for JSON.\n"
    )
    if profile:
        system_prompt += profile + "\n"
    if conversation_context:
        system_prompt += conversation_context + "\n"

    # --- Step 1: Conversational reply ---
    response = my_llm.invoke([
        {"role": "system", "content": system_prompt},
        *messages
    ])

    # --- Step 2: Memory extraction (silent) ---
    latest_user_message = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        ""
    )
    if latest_user_message.strip():
        try:
            manager.invoke({"messages": [{"role": "user", "content": latest_user_message}]})
        except Exception as e:
            print(f"[Memory extraction error] {e}")

    return response


# ----------------- FastAPI Setup -----------------
app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    messages: List[dict]

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # Load existing memory if available
    summary_manager.load_from_json(filepath=f"{request.user_id}_history.json")

    # Get LLM response
    response = await chat.ainvoke(
        request.messages,
        config={"configurable": {"user_id": request.user_id}}
    )

    # Update summary manager after responding
    user_msg = request.messages[-1]["content"]
    ai_msg = response.content if hasattr(response, "content") else str(response)
    summary_manager.update(user_msg, ai_msg)

    # Save memory
    summary_manager.save_to_json(filepath=f"{request.user_id}_history.json")

    return {"response": ai_msg}

@app.get("/memory/{user_id}")
async def get_memory(user_id: str):
    summary_manager.load_from_json(filepath=f"{user_id}_history.json")
    results = store.search(("users", user_id, "profile"))
    return {
        "profile": [r.value for r in results],
        "conversation_context": summary_manager.get_full_summary()
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
