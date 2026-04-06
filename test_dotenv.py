from dotenv import load_dotenv
import os
load_dotenv()
print("GROQ_API_KEY inside python:", repr(os.environ.get("GROQ_API_KEY")))
