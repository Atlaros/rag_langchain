from app.core.config import settings

class PromptManager:
    def __init__(self):
        self.system_prompt = settings.system_prompt
    def build_prompt(self, question, context):
        prompt = f"""{self.system_prompt}

Contexto relevante:
{context}

Pregunta:
{question}

Responde con diagnóstico breve, recomendaciones y ejemplos. Si falta información, indícalo claramente.""".strip()
        return prompt
