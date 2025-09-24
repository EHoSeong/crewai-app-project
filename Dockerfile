# 파이썬 3.12 버전의 경량 이미지를 베이스로 사용합니다.
FROM python:3.12-slim

# 컨테이너 내부의 작업 디렉토리를 /app으로 설정합니다.
WORKDIR /app

# 필요한 파이썬 패키지들을 설치합니다.
# requirements.txt 파일을 사용하지 않고 직접 나열하는 방식입니다.
RUN pip install --no-cache-dir \
    "crewai[tools]" \
    langchain_google_genai \
    python-dotenv \
    fastapi \
    uvicorn \
    pydantic \
    starlette

# 현재 디렉터리의 모든 파일을 컨테이너의 /app으로 복사합니다.
# main.py, index.html, .env 파일이 모두 포함됩니다.
COPY . .

# Uvicorn 서버를 시작하는 명령어입니다.
# --host 0.0.0.0 설정은 외부에서 접속 가능하게 해줍니다.
# --port 80은 HTTP 기본 포트입니다.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]