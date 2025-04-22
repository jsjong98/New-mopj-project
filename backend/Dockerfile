# GPU 지원을 위한 PyTorch 베이스 이미지 사용
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# 시스템 의존성 설치 (matplotlib 등에 필요한 라이브러리)
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 필요한 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 필요한 디렉토리 생성
RUN mkdir -p models static/plots static/reports static/ma_plots static/attention uploads

# 컨테이너 실행 시 서버 시작
EXPOSE 5000
CMD ["python", "app.py"]
