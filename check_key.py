import os
from dotenv import load_dotenv, dotenv_values, find_dotenv

path = find_dotenv()
file_key = dotenv_values(path).get("ANTHROPIC_API_KEY") if path else None
load_dotenv()
env_key = os.getenv("ANTHROPIC_API_KEY")

def peek(k):
    if not k:
        return None
    return f"prefix={k[:12]!r} len={len(k)} stray_space={k != k.strip()} quoted={k[0] in '\'\"'}"

print(".env file used :", path or "NOT FOUND")
print("key in .env    :", peek(file_key))
print("key in environ :", peek(env_key))
print("same           :", file_key == env_key)