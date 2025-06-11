# MOPJ 가격 예측 시스템 - 프론트엔드

## 📋 개요
React 기반 MOPJ 가격 예측 시스템의 사용자 인터페이스입니다. 인터랙티브한 차트와 직관적인 대시보드를 통해 예측 결과를 시각화합니다.

## 🔧 환경 요구사항
- Node.js 16.0 이상
- npm 8.0 이상 또는 yarn 1.22 이상
- 모던 웹 브라우저 (Chrome, Firefox, Safari, Edge)

## 📦 설치 방법

### 1. 의존성 패키지 설치
```bash
# npm 사용
npm install

# 또는 yarn 사용
yarn install
```

### 2. 환경 설정
```bash
# .env 파일 생성 (선택사항)
REACT_APP_API_URL=http://localhost:5000
REACT_APP_VERSION=1.0.0
```

## 🚀 실행 방법

### 개발 모드
```bash
# npm 사용
npm start

# 또는 yarn 사용
yarn start
```
개발 서버가 `http://localhost:3000`에서 시작됩니다.

### 프로덕션 빌드
```bash
# npm 사용
npm run build

# 또는 yarn 사용
yarn build
```

### 테스트 실행
```bash
# npm 사용
npm test

# 또는 yarn 사용
yarn test
```

## 🎯 주요 기능

### 1. 파일 업로드 인터페이스
- 드래그 앤 드롭으로 CSV 파일 업로드
- 파일 검증 및 미리보기
- 업로드 진행률 표시

### 2. 예측 설정 및 실행
- **단일 예측**: 날짜 선택기를 통한 예측 시작일 설정
- **누적 예측**: 시작일과 종료일 범위 설정
- 실시간 예측 진행률 모니터링

### 3. 결과 시각화
- **예측 차트**: Recharts 기반 인터랙티브 차트
- **이동평균 분석**: 다중 시간대 이동평균 비교
- **Attention 시각화**: 특성 중요도 히트맵
- **성능 지표**: 실시간 성능 메트릭 표시

### 4. 누적 분석 대시보드
- **날짜별 예측 비교**: 테이블 형태의 예측 결과 비교
- **추이 분석**: 누적 예측의 일관성 분석
- **구매 신뢰도**: 투자 결정 지원 지표

## 🏗️ 컴포넌트 구조

```
src/
├── App.js                      # 메인 애플리케이션 컴포넌트
├── components/                 # 재사용 가능한 컴포넌트
│   ├── PredictionChart.js     # 기본 예측 차트
│   ├── MovingAverageChart.js  # 이동평균 분석 차트
│   ├── AttentionMap.js        # Attention 가중치 시각화
│   ├── IntervalScoresTable.js # 구간 점수 테이블
│   ├── AccumulatedResultsTable.js # 누적 결과 테이블
│   ├── AccumulatedMetricsChart.js # 누적 지표 차트
│   ├── AccumulatedSummary.js  # 누적 예측 요약
│   ├── ProgressBar.js         # 진행률 표시 바
│   └── DateSelector.js        # 날짜 선택 컴포넌트
├── services/                   # API 서비스
│   └── api.js                 # 백엔드 API 호출
└── utils/                     # 유틸리티 함수
    └── formatting.js          # 데이터 포맷팅 함수
```

## 🎨 사용된 라이브러리

### 핵심 라이브러리
- **React 18.2.0**: UI 프레임워크
- **Recharts 2.15.2**: 차트 및 데이터 시각화
- **Lucide React 0.487.0**: 아이콘 라이브러리
- **Axios 1.8.4**: HTTP 클라이언트

### 유틸리티 라이브러리
- **React Modal 3.16.3**: 모달 다이얼로그
- **HTTP Proxy Middleware 3.0.5**: 개발 서버 프록시

### 테스팅 라이브러리
- **React Testing Library**: 컴포넌트 테스트
- **Jest DOM**: DOM 테스트 유틸리티

## 🎯 주요 상태 관리

### 전역 상태 (App.js)
```javascript
// 파일 및 데이터 상태
const [fileInfo, setFileInfo] = useState(null);
const [predictableStartDates, setPredictableStartDates] = useState([]);

// 예측 상태
const [isPredicting, setIsPredicting] = useState(false);
const [progress, setProgress] = useState(0);

// 결과 상태
const [predictionData, setPredictionData] = useState([]);
const [intervalScores, setIntervalScores] = useState([]);
const [maResults, setMaResults] = useState(null);
const [attentionImage, setAttentionImage] = useState(null);

// 누적 예측 상태
const [accumulatedResults, setAccumulatedResults] = useState(null);
const [selectedAccumulatedDate, setSelectedAccumulatedDate] = useState(null);
```

## 📊 API 통신

### 서비스 함수 (`services/api.js`)
```javascript
// 파일 업로드
export const uploadFile = (file) => { /* ... */ };

// 예측 실행
export const startPrediction = (filepath, currentDate) => { /* ... */ };
export const startAccumulatedPrediction = (filepath, startDate, endDate) => { /* ... */ };

// 결과 조회
export const getPredictionResults = () => { /* ... */ };
export const getAccumulatedResults = () => { /* ... */ };

// 상태 확인
export const getPredictionStatus = () => { /* ... */ };
```

## 🎨 스타일링

### 인라인 스타일 시스템
```javascript
const styles = {
  card: {
    backgroundColor: 'white',
    borderRadius: '0.5rem',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    padding: '1rem',
    marginBottom: '1.5rem'
  },
  // ... 기타 스타일
};
```

### 반응형 디자인
- **모바일 퍼스트**: 768px 브레이크포인트 기준
- **플렉시블 레이아웃**: CSS Flexbox 및 Grid 활용
- **적응형 차트**: 화면 크기에 따른 차트 크기 조정

## 🔧 개발 도구

### 디버깅
```javascript
// 콘솔 로그 시스템
console.log(`🔄 [FETCH] Starting fetchResults...`);
console.log(`✅ [STATE] States updated successfully`);
console.error(`❌ [ERROR] Prediction failed:`, error);
```

### 성능 모니터링
- React DevTools 호환
- 컴포넌트 렌더링 최적화
- 메모리 사용량 모니터링

## 🚀 배포

### 정적 파일 생성
```bash
npm run build
```

### 배포 옵션
1. **Netlify**: `build` 폴더를 드래그 앤 드롭
2. **Vercel**: GitHub 연동 자동 배포
3. **AWS S3**: S3 버킷에 정적 호스팅
4. **Nginx**: 역프록시 설정으로 백엔드와 통합

## 🐛 문제 해결

### 1. 패키지 설치 오류
```bash
# npm 캐시 정리
npm cache clean --force

# node_modules 재설치
rm -rf node_modules package-lock.json
npm install
```

### 2. 프록시 연결 오류
```bash
# 백엔드 서버 실행 확인
curl http://localhost:5000/api/health

# package.json proxy 설정 확인
"proxy": "http://localhost:5000"
```

### 3. 차트 렌더링 문제
- 브라우저 콘솔에서 JavaScript 오류 확인
- Recharts 버전 호환성 확인
- 데이터 구조 검증

## 📞 지원
프론트엔드 관련 문제 발생 시 브라우저 개발자 도구(F12)의 콘솔 탭에서 오류 메시지를 확인하고 개발팀에 문의하세요.

---
© 2024 MOPJ 가격 예측 시스템
