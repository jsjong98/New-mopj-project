# GPU 지원 PyTorch 베이스 이미지
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 비대화형 설치 + 타임존 설정
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

WORKDIR /app

# 시스템 의존성 설치 (tzdata + distutils 포함)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tzdata \
      build-essential \
      libopenblas-dev \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender-dev \
      python3-distutils \
      python3-dev \
 && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
 && echo $TZ > /etc/timezone \
 && dpkg-reconfigure -f noninteractive tzdata \
 && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 로그·모델 등 필요한 디렉터리 미리 생성
RUN mkdir -p models static/plots static/reports static/ma_plots static/attention uploads

EXPOSE 5000
CMD ["python", "app.py"]
