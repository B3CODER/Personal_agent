from fastapi import FastAPI, Body
from pydantic import BaseModel
import uvicorn

from langchain.chat_models import init_chat_model
from langgraph.func import entrypoint
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_config
from langmem import create_memory_store_manager
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel as PydanticBaseModel, field_validator
from typing import Optional, List
import os
from dotenv import load_dotenv
load_dotenv()

# ----------------- Your Existing Models -----------------
class UserProfile(PydanticBaseModel):
    tone: Optional[str] = None
    words: Optional[str] = None
    audience: Optional[str] = None
    format_structure: Optional[str] = None
    content_type: Optional[str] = None
    engagement: Optional[str] = None

    @field_validator("audience", mode="before")
    @classmethod
    def parse_audience(cls, v):
        if isinstance(v, str):
            import json
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed[0] if parsed else None
            except:
                return v


embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_fn(texts):
    return embed_model.encode(texts, convert_to_numpy=True).tolist()


store = InMemoryStore(
    index={
        "dims": 384,
        "embed": embed_fn,
    }
)

manager = create_memory_store_manager(
    "google_genai:gemini-2.5-flash",
    namespace=("users", "{user_id}", "profile"),
    schemas=[UserProfile],
    instructions="Extract all user information and events as triples.",
    enable_inserts=True,
    enable_deletes=True,
)

my_llm = init_chat_model("google_genai:gemini-2.5-flash")


@entrypoint(store=store)
def chat(messages: list):
    configurable = get_config()["configurable"]
    results = store.search(
        ("users", configurable["user_id"], "profile")
    )
    profile = None
    if results:
        profile = f"""<User Profile>:

{results[0].value}
</User Profile>
"""

    response = my_llm.invoke([
        {
            "role": "system",
            "content": f"""You are a helpful assistant.{profile}"""
        },
        *messages
    ])

    # Update profile with any new information
    manager.invoke({"messages": messages})
    return response


# ----------------- FastAPI Setup -----------------
app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    messages: List[dict]  # [{"role": "user", "content": "Hello"}]

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    response = await chat.ainvoke(
        request.messages,
        config={"configurable": {"user_id": request.user_id}}
    )
    return {"response": response}

@app.get("/memory/{user_id}")
async def get_memory(user_id: str):
    results = store.search(("users", user_id, "profile"))
    return {"profile": [r.value for r in results]}
# ----------------- Run Server -----------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
