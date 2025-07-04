# GPU 지원 PyTorch 베이스 이미지
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# 비대화형 설치 + 타임존 설정
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

WORKDIR /app

# 시스템 의존성 (tzdata + distutils + dev 헤더)
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

# 1) pip/​setuptools/​wheel 최신화
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 2) 주요 과학 스택을 conda로 설치 (C-API 버전 충돌 방지)
RUN conda install -y \
      numpy=1.24.3 \
      pandas=2.1.1 \
      scikit-learn=1.3.0 \
      matplotlib=3.7.2 \
      seaborn=0.12.2 \
      python-dateutil=2.8.2 \
      pytz=2023.3 \
 && conda clean -afy

# 3) Docker 전용 requirements로 나머지 패키지 설치
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# 애플리케이션 코드 복사
COPY . .

# 로그·모델·업로드 폴더 미리 생성
RUN mkdir -p models static/plots static/reports static/ma_plots static/attention uploads

EXPOSE 5000
CMD ["python", "app.py"]
