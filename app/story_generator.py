import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class StoryGenerator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Please set it in your environment or .env.")
        self.client = Groq(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    def generate_story(
        self,
        name: str,
        age: int,
        genre: str,
        language: str = "en",
        custom_prompt: str = None,
        attention_span: str = "medium",   # "short", "medium", "long"
        reading_level: str = "basic"      # "basic", "intermediate", "advanced"
    ) -> str:
        if custom_prompt and custom_prompt.strip():
            prompt = custom_prompt
            max_tokens = 300
        else:
            # Adjust token limits for different languages - non-English languages need more tokens
            # because they use more characters/bytes for the same semantic content
            # Telugu and Hindi need even more tokens due to complex script
            if language in ["te", "hi"]:
                token_multiplier = 2.5
            elif language in ["es", "fr", "de"]:
                token_multiplier = 1.8
            else:
                token_multiplier = 1.0

            if attention_span == "short":
                length_desc = "a very short story"
                max_tokens = int(250 * token_multiplier)
            elif attention_span == "long":
                length_desc = "a longer, more detailed story"
                max_tokens = int(800 * token_multiplier)
            else:
                length_desc = "a complete story with beginning, middle, and end"
                max_tokens = int(600 * token_multiplier)

            if reading_level == "basic":
                complexity_desc = "written in simple, easy-to-understand language"
            elif reading_level == "advanced":
                complexity_desc = "with rich vocabulary and complex sentence structure"
            else:
                complexity_desc = "with clear language and moderate complexity"

            # Map language codes to full language names for better LLM understanding
            language_names = {
                "en": "English",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
                "te": "Telugu",
                "hi": "Hindi"
            }
            lang_name = language_names.get(language, language)

            if language == "te":
                # Special handling for Telugu - use more explicit instructions with structured format
                prompt = (
                    f"తెలుగు భాషలో మాత్రమే సంపూర్ణంగా రాయండి. {age}-సంవత్సరాల వయస్సు గల {name} పేరు గల పిల్లకు "
                    f"{genre} జాతీయంలో {length_desc} రాయండి. \n\n"
                    f"కథను ఈ విధంగా నిర్మించండి:\n"
                    f"1. మొదటి భాగం: కథ ప్రారంభం మరియు పాత్రల పరిచయం\n"
                    f"2. మధ్య భాగం: సమస్య లేదా సంఘటన\n"
                    f"3. ముగింపు భాగం: సమస్య పరిష్కారం మరియు నీతి పాఠం\n\n"
                    f"కథ చివరిలో తప్పకుండా నీతి పాఠం చెప్పండి. బెడ్‌టైమ్ కోసం సానుకూల, శాంత సందేశంతో ముగించండి. "
                    f"కథను సంపూర్ణంగా తెలుగులో మాత్రమే రాయండి, ఎటువంటి ఇంగ్లీష్ టెక్స్ట్ ఉండకూడదు."
                )
            else:
                prompt = (
                    f"Write {length_desc} entirely in {lang_name} language for a {age}-year-old child named {name}. "
                    f"The story should be in the {genre} genre, {complexity_desc}, "
                    f"have a clear beginning, middle, and conclusion with a moral lesson. "
                    f"End with a positive, calming message for bedtime. "
                    f"IMPORTANT: Respond ONLY in {lang_name}, do not include any English text."
                )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a friendly story generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_completion_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
