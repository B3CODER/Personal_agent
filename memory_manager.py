import json
import os
from langchain_core.prompts import PromptTemplate

class SummaryManager:
    def __init__(self, llm, summary_prompt, max_chunks=5):
        self.llm = llm
        self.summary_prompt = summary_prompt
        self.max_chunks = max_chunks

        self.message_counter = 0
        self.window_messages = []      # Holds latest 10 messages
        self.chunk_summaries = []      # Holds 10-message summaries
        self.final_summaries = []      # Holds 50-message merged summaries

    def _summarize_chunk(self, messages):
        try:
            new_lines = "\n".join([f"User: {m['input']}\nAI: {m['output']}" for m in messages])
            prompt = self.summary_prompt.format(summary="", new_lines=new_lines)
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"[ERROR] Failed to summarize chunk: {e}")
            return "[ERROR] Summary generation failed."

    def _merge_summaries(self):
        try:
            combined_summary = "\n".join(self.chunk_summaries)
            prompt = self.summary_prompt.format(summary="", new_lines=combined_summary)
            response = self.llm.invoke(prompt)
            merged_summary = response.content if hasattr(response, 'content') else str(response)

            self.final_summaries.append(merged_summary)

            # Keep only last 3 merged summaries
            if len(self.final_summaries) > 3:
                self.final_summaries.pop(0)

            self.chunk_summaries = []

        except Exception as e:
            print(f"[ERROR] Failed to merge summaries: {e}")
            self.chunk_summaries = []  # Reset anyway to prevent loop

    def update(self, input_text, output_text):
        try:
            self.message_counter += 1
            self.window_messages.append({"input": input_text, "output": output_text})

            if len(self.window_messages) == 5:  # change to 10 in production
                summary = self._summarize_chunk(self.window_messages)
                self.chunk_summaries.append(summary)
                self.window_messages = []

                if len(self.chunk_summaries) == self.max_chunks:
                    self._merge_summaries()

        except Exception as e:
            print(f"[ERROR] Failed to update conversation memory: {e}")

    def get_full_summary(self):
        return "\n\n".join(self.final_summaries + self.chunk_summaries)

    def get_recent_messages(self):
        return self.window_messages

    def save_to_json(self, filepath="conversation_memory.json"):
        try:
            data = {
                "message_counter": self.message_counter,
                "window_messages": self.window_messages,
                "chunk_summaries": self.chunk_summaries,
                "final_summaries": self.final_summaries
            }
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Memory successfully saved to {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to save memory to {filepath}: {e}")

    def load_from_json(self, filepath="conversation_memory.json"):
        if not os.path.exists(filepath):
            print(f"[WARNING] File '{filepath}' not found. Starting fresh.")
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.message_counter = data.get("message_counter", 0)
            self.window_messages = data.get("window_messages", [])
            self.chunk_summaries = data.get("chunk_summaries", [])
            self.final_summaries = data.get("final_summaries", [])

            print(f"[INFO] Memory successfully loaded from {filepath}")
        except json.JSONDecodeError:
            print(f"[ERROR] Failed to parse JSON in {filepath}. File may be corrupted.")
        except Exception as e:
            print(f"[ERROR] Failed to load memory from {filepath}: {e}")
