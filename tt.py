import os
from dotenv import load_dotenv

# .env 파일을 불러와 환경 변수로 설정
load_dotenv()

# 변수가 잘 로드되었는지 확인
gemini_key = os.getenv("GOOGLE_API_KEY")
serper_key = os.getenv("SERPER_API_KEY")

print(f"Gemini API Key: {gemini_key}")
print(f"Serper API Key: {serper_key}")