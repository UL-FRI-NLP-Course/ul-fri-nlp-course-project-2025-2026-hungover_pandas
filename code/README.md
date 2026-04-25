## Setup

1. Install dependencies:
   pip install -r requirements.txt

2. Copy .env.example to .env and fill in your API key:
   cp .env.example .env
   (then open .env and paste your Gemini API key)

   Get a free Gemini API key at: https://aistudio.google.com/apikey

3. Build the index:
   python index.py

4. Run the chatbot:
   python rag.py