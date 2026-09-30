import requests


class OllamaClient:
    """Client for a local Ollama LLM server."""

    def __init__(self, model_name: str = "phi3", base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate(self, prompt: str, retries: int = 3) -> str:
        if self.model_name.startswith("qwen3"):
            return self._chat_generate(prompt)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 512,
            },
        }

        last_error = None
        for attempt in range(retries):
            try:
                response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=300)
                response.raise_for_status()
                data = response.json()
                answer = data.get("response", "").strip()

                if not answer:
                    raise RuntimeError(
                        "Ollama returned no final answer. The model may still be thinking, "
                        "or your laptop may need a smaller/faster model such as phi3."
                    )

                return answer
            except Exception as error:
                last_error = error
                if attempt < retries - 1:
                    import time
                    time.sleep(2)

        assert last_error is not None
        raise last_error

    def _chat_generate(self, prompt: str) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are AI Study Buddy. Give only the final answer. "
                        "Do not include thinking, reasoning traces, or hidden analysis."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 768,
            },
        }

        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=300)
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        answer = message.get("content", "").strip()
        answer = self._clean_qwen_answer(answer)

        if not answer:
            raise RuntimeError(
                "Ollama returned no final answer from the chat endpoint. "
                "Try `ollama pull phi3` for a faster local model."
            )

        return answer

    def _clean_qwen_answer(self, answer: str) -> str:
        markers = ["Answer:", "Final answer:", "Final:"]
        for marker in markers:
            if marker in answer:
                return answer.split(marker, 1)[1].strip()

        if "<think>" in answer and "</think>" in answer:
            return answer.split("</think>", 1)[1].strip()

        reasoning_phrases = [
            "the user is asking",
            "first, i need",
            "i need to",
            "let me",
            "i should",
        ]
        lowered = answer.lower()
        if any(phrase in lowered for phrase in reasoning_phrases):
            lines = [line.strip() for line in answer.splitlines() if line.strip()]
            useful_lines = [
                line for line in lines
                if not any(phrase in line.lower() for phrase in reasoning_phrases)
            ]
            if useful_lines:
                return useful_lines[-1]
            return ""

        return answer
