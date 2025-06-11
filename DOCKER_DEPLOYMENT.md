# 🐳 Docker 배포 가이드

## 📋 개요
MOPJ 가격 예측 시스템의 Docker 컨테이너 배포 가이드입니다. GPU 지원 환경과 CPU 전용 환경 모두에 대한 설정을 제공합니다.

## 🔧 환경 요구사항

### 최소 요구사항
- Docker 20.10 이상
- Docker Compose 2.0 이상
- 최소 8GB RAM
- 최소 10GB 디스크 공간

### GPU 지원 (권장)
- NVIDIA GPU 드라이버 450.80.02 이상
- NVIDIA Container Toolkit
- CUDA 11.7 이상

## 🚀 빠른 시작

### 1. GPU 지원 환경 (권장)

#### NVIDIA Container Toolkit 설치
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### 시스템 실행
```bash
# GPU 지원 docker-compose 사용
docker-compose up -d

# 로그 확인
docker-compose logs -f
```

### 2. CPU 전용 환경

```bash
# CPU 전용 docker-compose 사용
docker-compose -f docker-compose.cpu.yml up -d

# 로그 확인
docker-compose -f docker-compose.cpu.yml logs -f
```

## 📊 서비스 구성

### 백엔드 (mopj-backend)
- **포트**: 5000
- **이미지**: 커스텀 빌드 (PyTorch + Flask)
- **볼륨**: 모델, 캐시, 업로드 파일 영구 저장
- **헬스체크**: `/api/health` 엔드포인트

### 프론트엔드 (mopj-frontend)
- **포트**: 80, 443
- **이미지**: Nginx + React 빌드
- **프록시**: 백엔드 API 요청 자동 라우팅
- **헬스체크**: HTTP 응답 확인

## 🔍 상태 확인

### 컨테이너 상태 확인
```bash
# 실행 중인 컨테이너 확인
docker-compose ps

# 헬스체크 상태 확인
docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"
```

### 로그 모니터링
```bash
# 전체 로그
docker-compose logs

# 특정 서비스 로그
docker-compose logs backend
docker-compose logs frontend

# 실시간 로그 추적
docker-compose logs -f --tail=100
```

### 리소스 사용량 확인
```bash
# 컨테이너 리소스 사용량
docker stats

# GPU 사용량 (GPU 환경)
nvidia-smi
```

## 🔧 환경 설정

### 환경 변수 설정
```bash
# .env 파일 생성
cat > .env << EOF
# 백엔드 설정
FLASK_ENV=production
PYTHONUNBUFFERED=1

# GPU 설정 (필요한 경우)
NVIDIA_VISIBLE_DEVICES=all
CUDA_VISIBLE_DEVICES=0,1

# 로그 레벨
LOG_LEVEL=INFO
EOF
```

### 포트 변경
```yaml
# docker-compose.yml에서 포트 수정
services:
  frontend:
    ports:
      - "8080:80"  # 포트 80 대신 8080 사용
  backend:
    ports:
      - "5001:5000"  # 포트 5000 대신 5001 사용
```

## 💾 데이터 영속성

### 볼륨 관리
```bash
# 볼륨 목록 확인
docker volume ls

# 볼륨 상세 정보
docker volume inspect new-mopj-project-main_backend_models

# 백업 생성
docker run --rm -v new-mopj-project-main_backend_models:/data -v $(pwd):/backup alpine tar czf /backup/models-backup.tar.gz /data
```

### 데이터 마이그레이션
```bash
# 볼륨 데이터 복원
docker run --rm -v new-mopj-project-main_backend_models:/data -v $(pwd):/backup alpine tar xzf /backup/models-backup.tar.gz -C /
```

## 🔄 업데이트 및 재배포

### 1. 이미지 업데이트
```bash
# 이미지 다시 빌드
docker-compose build --no-cache

# 서비스 재시작
docker-compose up -d
```

### 2. 코드 변경 후 배포
```bash
# 개발 중인 변경사항 반영
docker-compose down
docker-compose build
docker-compose up -d
```

### 3. 롤링 업데이트
```bash
# 백엔드만 업데이트
docker-compose up -d --no-deps backend

# 프론트엔드만 업데이트
docker-compose up -d --no-deps frontend
```

## 🐛 문제 해결

### 1. 컨테이너 실행 오류
```bash
# 상세 로그 확인
docker-compose logs backend

# 컨테이너 내부 접속
docker exec -it mopj-backend bash

# 디스크 용량 확인
df -h
docker system df
```

### 2. GPU 인식 오류
```bash
# GPU 상태 확인
nvidia-smi

# NVIDIA Docker 런타임 확인
docker run --rm --gpus all nvidia/cuda:11.7-base-ubuntu20.04 nvidia-smi

# 컨테이너 내 GPU 확인
docker exec -it mopj-backend nvidia-smi
```

### 3. 메모리 부족
```bash
# 메모리 사용량 확인
docker stats --no-stream

# 불필요한 이미지 정리
docker system prune -a

# 볼륨 정리
docker volume prune
```

### 4. 네트워크 연결 문제
```bash
# 네트워크 상태 확인
docker-compose exec frontend curl -f http://backend:5000/api/health

# 포트 충돌 확인
netstat -tulpn | grep :80
netstat -tulpn | grep :5000
```

## 📈 성능 최적화

### 1. 리소스 제한 설정
```yaml
# docker-compose.yml에 추가
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'
```

### 2. 이미지 크기 최적화
```bash
# 이미지 크기 확인
docker images | grep mopj

# 빌드 캐시 최적화
docker-compose build --build-arg BUILDKIT_INLINE_CACHE=1
```

### 3. 볼륨 성능 향상
```bash
# SSD 사용 확인
lsblk -d -o name,rota

# 볼륨 마운트 옵션 최적화 (docker-compose.yml)
volumes:
  - backend_models:/app/models:cached
```

## 🔐 보안 설정

### 1. 네트워크 격리
```yaml
# docker-compose.yml에 커스텀 네트워크 추가
networks:
  mopj-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

services:
  backend:
    networks:
      - mopj-network
```

### 2. 환경 변수 보안
```bash
# 민감한 정보는 Docker secrets 사용
echo "secret_key_here" | docker secret create flask_secret_key -
```

## 📞 지원

### 로그 수집
```bash
# 전체 시스템 정보 수집
docker-compose logs > deployment-logs.txt
docker system info > system-info.txt
docker-compose ps --format json > container-status.json
```

### 디버깅 모드
```bash
# 개발 모드로 실행
FLASK_ENV=development docker-compose up
```

---
© 2025 MOPJ 가격 예측 시스템 
