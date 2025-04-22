# GPU 지원 PyTorch 이미지
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 1) 비대화형 모드 설정  2) 기본 타임존 환경변수 지정
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

WORKDIR /app

# 시스템 의존성 설치 (비대화형, tzdata 포함)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tzdata \
      build-essential \
      libopenblas-dev \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev \
 && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && dpkg-reconfigure -f noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY . .

# 필요한 디렉터리 생성
RUN mkdir -p models static/plots static/reports static/ma_plots static/attention uploads

EXPOSE 5000
CMD ["python", "app.py"]
