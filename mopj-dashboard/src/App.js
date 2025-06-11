import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, Calendar, Database, Clock, Grid, Award, RefreshCw, AlertTriangle, BarChart, Activity
} from 'lucide-react';
import FileUploader from './components/FileUploader';
import PredictionChart from './components/PredictionChart';
import MovingAverageChart from './components/MovingAverageChart';
import IntervalScoresTable from './components/IntervalScoresTable';
import AttentionMap from './components/AttentionMap';
import ProgressBar from './components/ProgressBar';
import AccumulatedMetricsChart from './components/AccumulatedMetricsChart';
import AccumulatedResultsTable from './components/AccumulatedResultsTable';
import AccumulatedSummary from './components/AccumulatedSummary';
import ReliabilityAnalysisCard from './components/ReliabilityAnalysisCard';
import AccumulatedIntervalScoresTable from './components/AccumulatedIntervalScoresTable';
import HolidayManager from './components/HolidayManager';
import CalendarDatePicker from './components/CalendarDatePicker'; // 달력 컴포넌트 추가
import { 
  startPrediction, 
  getPredictionStatus, 
  getPredictionResults,
  startAccumulatedPrediction,
  getAccumulatedResults,
  getAccumulatedResultByDate,
  getAccumulatedReportURL,
  checkCachedPredictions,
  clearAccumulatedCache,
  getRecentAccumulatedResults,
  getHolidays,
  getAttentionMap
} from './services/api';

// Helper 함수들 (예측 시작일 방식) - 수정됨

// 휴일 체크 함수
const isHoliday = (dateString, holidays) => {
  return holidays.some(holiday => holiday.date === dateString);
};

// ✅ isBusinessDay 함수 제거 (사용되지 않음)

const getNextBusinessDay = (dateString, holidays = []) => {
  // UTC 기준으로 날짜 생성하여 타임존 이슈 방지
  const [year, month, day] = dateString.split('-').map(Number);
  const date = new Date(year, month - 1, day); // month는 0-based
  
  date.setDate(date.getDate() + 1);
  
  // 주말이거나 휴일이면 다음 영업일까지 이동
  // 0=일요일, 6=토요일
  while (date.getDay() === 0 || date.getDay() === 6 || isHoliday(formatDateYMD(date), holidays)) {
    date.setDate(date.getDate() + 1);
  }
  
  // YYYY-MM-DD 형식으로 반환
  const year2 = date.getFullYear();
  const month2 = String(date.getMonth() + 1).padStart(2, '0');
  const day2 = String(date.getDate()).padStart(2, '0');
  return `${year2}-${month2}-${day2}`;
};

// Date 객체를 YYYY-MM-DD 형식으로 변환하는 헬퍼 함수
const formatDateYMD = (date) => {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
};

// ✅ getPreviousBusinessDay 함수 제거 (사용되지 않음)

const formatDate = (dateString) => {
  // 타임존 이슈 방지를 위해 로컬 날짜로 파싱
  const [year, month, day] = dateString.split('-').map(Number);
  const date = new Date(year, month - 1, day);
  
  return date.toLocaleDateString('ko-KR', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    weekday: 'long'
  });
};

// 반월 기간의 시작일인지 확인하는 함수
const isSemimonthlyStart = (dateString) => {
  const date = new Date(dateString + 'T00:00:00');
  const day = date.getDate();
  // 1일 또는 16일이면 반월 시작
  return day === 1 || day === 16;
};

// ✅ getNextSemimonthlyStart 함수 제거 (사용되지 않음)

// 예측 가능한 시작일 목록 생성 (데이터의 50% 지점부터, 반월 기준 우선)
const generatePredictableStartDates = (dataDatesList, holidays = []) => {
  if (!Array.isArray(dataDatesList) || dataDatesList.length === 0) {
    return [];
  }
  
  console.log(`🔍 [DATE_GENERATION] Processing ${dataDatesList.length} data dates with ${holidays.length} holidays`);
  console.log(`🔍 [DATE_GENERATION] Sample data dates:`, dataDatesList.slice(0, 5));
  console.log(`🔍 [DATE_GENERATION] Sample holidays:`, holidays.slice(0, 5).map(h => h.date || h));
  
  console.log(`📊 [DATA_INFO] Total dates from backend: ${dataDatesList.length}`);
  console.log(`📊 [DATA_INFO] Backend already filtered 50%+ data: ${dataDatesList[0]} ~ ${dataDatesList[dataDatesList.length - 1]}`);
  
  // 🎯 백엔드에서 이미 50% 필터링된 데이터를 받았으므로 모든 날짜를 예측 가능으로 처리
  const validStartDates = [];
  
  dataDatesList.forEach((dataDate, index) => {
    // 해당 데이터가 있으면 그 다음 영업일을 예측 시작일로 표시
    const nextBusinessDay = getNextBusinessDay(dataDate, holidays);
    
    validStartDates.push({
      startDate: nextBusinessDay,
      requiredDataDate: dataDate, // 예측에 필요한 데이터 마지막 날짜
      label: formatDate(nextBusinessDay),
      isHoliday: isHoliday(nextBusinessDay, holidays),
      isSemimonthlyStart: isSemimonthlyStart(nextBusinessDay), // 반월 시작 여부
      dataIndex: index // 전체 데이터에서의 인덱스
    });
    
    console.log(`✅ [DATE_GENERATION] Added: ${nextBusinessDay} (uses data until: ${dataDate}, index: ${index}, semimonthly: ${isSemimonthlyStart(nextBusinessDay)})`);
  });
  
  // 중복 제거 및 반월 시작일 우선 처리
  const uniqueStartDates = [];
  const seenStartDates = new Map(); // startDate -> { requiredDataDate, index, isSemimonthlyStart }
  
  // 데이터 날짜 순서를 유지하면서 중복 제거 (반월 시작일 우선)
  validStartDates.forEach((item, index) => {
    if (!seenStartDates.has(item.startDate)) {
      // 첫 번째로 나온 경우 추가
      seenStartDates.set(item.startDate, { 
        requiredDataDate: item.requiredDataDate, 
        index,
        isSemimonthlyStart: item.isSemimonthlyStart 
      });
      uniqueStartDates.push(item);
      console.log(`📋 [DATE_FILTER] First occurrence: ${item.startDate} (uses data until: ${item.requiredDataDate}, semimonthly: ${item.isSemimonthlyStart})`);
    } else {
      // 같은 예측 시작일이 있다면 처리 우선순위: 1) 반월 시작일 2) 더 최근 데이터
      const existing = seenStartDates.get(item.startDate);
      let shouldReplace = false;
      
      if (!existing.isSemimonthlyStart && item.isSemimonthlyStart) {
        // 기존이 반월 시작일이 아니고 새 항목이 반월 시작일이면 교체
        shouldReplace = true;
        console.log(`🎯 [DATE_FILTER] Replacing with semimonthly start: ${item.startDate}`);
      } else if (existing.isSemimonthlyStart === item.isSemimonthlyStart && item.requiredDataDate > existing.requiredDataDate) {
        // 둘 다 반월 시작일이거나 둘 다 아닌 경우, 더 최근 데이터 우선
        shouldReplace = true;
        console.log(`🔄 [DATE_FILTER] Replacing with more recent data: ${item.startDate}`);
      }
      
      if (shouldReplace) {
        const existingIndex = uniqueStartDates.findIndex(existing => existing.startDate === item.startDate);
        if (existingIndex !== -1) {
          uniqueStartDates[existingIndex] = item;
          seenStartDates.set(item.startDate, { 
            requiredDataDate: item.requiredDataDate, 
            index,
            isSemimonthlyStart: item.isSemimonthlyStart 
          });
        }
      } else {
        console.log(`⚠️ [DATE_FILTER] Skipped: ${item.startDate} (existing has priority)`);
      }
    }
  });
  
  // 반월 시작일을 앞쪽으로 정렬 (우선 표시)
  uniqueStartDates.sort((a, b) => {
    // 날짜 순서는 유지하되, 같은 날짜라면 반월 시작일이 우선
    if (a.startDate === b.startDate) {
      return b.isSemimonthlyStart - a.isSemimonthlyStart;
    }
    return a.startDate.localeCompare(b.startDate);
  });
  
  console.log(`📋 [DATE_GENERATION] Generated ${uniqueStartDates.length} unique start dates from ${dataDatesList.length} prediction-eligible dates`);
  console.log(`📋 [DATE_GENERATION] Semimonthly starts: ${uniqueStartDates.filter(d => d.isSemimonthlyStart).length}`);
  console.log(`📋 [DATE_GENERATION] First 5 start dates:`, uniqueStartDates.slice(0, 5).map(item => `${item.startDate} (uses data until ${item.requiredDataDate}, semimonthly: ${item.isSemimonthlyStart})`));
  console.log(`📋 [DATE_GENERATION] Last 5 start dates:`, uniqueStartDates.slice(-5).map(item => `${item.startDate} (uses data until ${item.requiredDataDate}, semimonthly: ${item.isSemimonthlyStart})`));
  
  return uniqueStartDates;
};

// CSS 스타일 추가
const dropdownCSS = `
  .dropdown.show {
    opacity: 1 !important;
    visibility: visible !important;
    transform: translateY(0) !important;
  }
  
  .dropdown-item:hover {
    background-color: #f9fafb !important;
  }
`;

const App = () => {
  // 기본 상태 관리
  const [fileInfo, setFileInfo] = useState(null);
  const [selectedStartDate, setSelectedStartDate] = useState(null); // 예측 시작일
  const [endStartDate, setEndStartDate] = useState(null); // 누적 예측 종료 시작일
  const [requiredDataDate, setRequiredDataDate] = useState(null); // 필요한 데이터 기준일
  const [predictableStartDates, setPredictableStartDates] = useState([]); // 예측 가능한 시작일 목록
  const [isLoading, setIsLoading] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [currentDate, setCurrentDate] = useState(null);
  const [predictionData, setPredictionData] = useState([]);
  const [intervalScores, setIntervalScores] = useState([]);
  const [maResults, setMaResults] = useState(null);
  const [attentionImage, setAttentionImage] = useState(null);
  const [isCSVUploaded, setIsCSVUploaded] = useState(false);
  
  // 반응형 처리를 위한 state 추가
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  
  // 탭 관리
  const [activeTab, setActiveTab] = useState('single');
  
  // 시스템 탭 관리
  const [systemTab, setSystemTab] = useState('home');
  
  // 누적 예측 관련 상태
  const [accumulatedResults, setAccumulatedResults] = useState(null);
  const [selectedAccumulatedDate, setSelectedAccumulatedDate] = useState(null);

  // 선택된 날짜의 예측 결과 상태 (기본 상태와 분리)
  const [selectedDatePredictions, setSelectedDatePredictions] = useState([]);
  const [selectedDateIntervalScores, setSelectedDateIntervalScores] = useState([]);

  // ✅ 선택된 날짜 변화 모니터링
  useEffect(() => {
    if (selectedAccumulatedDate) {
      console.log(`🎯 [EFFECT] selectedAccumulatedDate changed to: ${selectedAccumulatedDate}`);
      console.log(`🎯 [EFFECT] Current selectedDatePredictions: ${selectedDatePredictions.length} items`);
      console.log(`🎯 [EFFECT] Current selectedDateIntervalScores: ${selectedDateIntervalScores.length} items`);
    }
  }, [selectedAccumulatedDate, selectedDatePredictions, selectedDateIntervalScores]);

  // 신뢰도 관련 상태
  const [consistencyScores, setConsistencyScores] = useState(null);

  // 캐시 정보 상태
  const [cacheInfo, setCacheInfo] = useState(null);
  
  // 휴일 정보 상태
  const [holidays, setHolidays] = useState([]);

  // 반응형 처리를 위한 useEffect 추가
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // 휴일 정보 로드
  useEffect(() => {
    const loadHolidays = async () => {
      try {
        const result = await getHolidays();
        if (result.success && result.holidays) {
          setHolidays(result.holidays);
          console.log('✅ [HOLIDAYS] Loaded holidays:', result.holidays.length);
          console.log('📅 [HOLIDAYS] Holiday dates:', result.holidays.map(h => h.date).slice(0, 10));
        }
      } catch (error) {
        console.error('❌ [HOLIDAYS] Failed to load holidays:', error);
      }
    };
    
    loadHolidays();
  }, []);

  // ✅ 페이지 로드시 자동으로 attention map 로딩 시도
  useEffect(() => {
    console.log('🚀 [INIT] Page loaded, trying to auto-load attention map...');
    loadAttentionMapAuto();
  }, []);

  // 페이지 로드 시 최근 누적 예측 결과 자동 복원 - 비활성화
  useEffect(() => {
    // 자동 복원 기능을 비활성화합니다. 사용자가 직접 파일을 업로드하고 예측을 실행해야 합니다.
    console.log('ℹ️ [AUTO_RESTORE] Auto-restore feature disabled. Please upload file and run prediction manually.');
  }, []); // 컴포넌트 마운트 시에만 실행

  // 예측 시작일이 변경될 때 필요한 데이터 기준일 계산
  useEffect(() => {
    if (selectedStartDate) {
      // 선택된 예측 시작일에 해당하는 필요한 데이터 기준일 찾기
      const selectedPrediction = predictableStartDates.find(p => p.startDate === selectedStartDate);
      if (selectedPrediction) {
        setRequiredDataDate(selectedPrediction.requiredDataDate);
      }
    }
  }, [selectedStartDate, predictableStartDates]);

  // 누적 예측 날짜가 변경될 때마다 캐시 정보 확인
  useEffect(() => {
    const checkCache = async () => {
      if (selectedStartDate && endStartDate && predictableStartDates.length > 0) {
        // 시작일과 종료일에 해당하는 데이터 기준일 찾기
        const startPredictableDate = predictableStartDates.find(p => p.startDate === selectedStartDate);
        const endPredictableDate = predictableStartDates.find(p => p.startDate === endStartDate);
        
        if (startPredictableDate && endPredictableDate) {
          try {
            const result = await checkCachedPredictions(
              startPredictableDate.requiredDataDate, 
              endPredictableDate.requiredDataDate
            );
            
            if (result.success) {
              setCacheInfo(result);
            } else {
              setCacheInfo(null);
            }
          } catch (err) {
            console.error('Cache check failed:', err);
            setCacheInfo(null);
          }
        }
      } else {
        setCacheInfo(null);
      }
    };
    
    checkCache();
  }, [selectedStartDate, endStartDate, predictableStartDates]);

  // 누적 예측 미리보기 계산 함수
  const calculateAccumulatedPreview = (startDate, endDate) => {
    if (!startDate || !endDate) return null;
    
    // 시작일과 종료일에 해당하는 예측 설정 찾기
    const startPredictableDate = predictableStartDates.find(p => p.startDate === startDate);
    const endPredictableDate = predictableStartDates.find(p => p.startDate === endDate);
    
    if (!startPredictableDate || !endPredictableDate) return null;
    
    // 범위 내의 모든 예측 가능한 날짜들 찾기
    const predictionDates = predictableStartDates.filter(p => 
      p.startDate >= startDate && p.startDate <= endDate
    );
    
    return {
      predictionCount: predictionDates.length,
      firstPredictionStart: startDate,
      lastPredictionStart: endDate,
      firstRequiredData: startPredictableDate.requiredDataDate,
      lastRequiredData: endPredictableDate.requiredDataDate,
      predictionDates
    };
  };

  // 핸들러 함수
  const handleUploadSuccess = (data) => {
    setFileInfo(data);
    
    // 🎯 캐시 정보 표시
    if (data.cache_info && data.cache_info.found) {
      const cacheMessage = data.cache_info.message;
      console.log(`✅ [CACHE] ${cacheMessage}`);
      
      // 사용자에게 캐시 정보 알림 (선택적)
      if (data.cache_info.cache_type === 'exact') {
        console.log('🎉 [CACHE] Exact match - predictions will be much faster!');
      } else if (data.cache_info.cache_type === 'extension') {
        const extInfo = data.cache_info.extension_info;
        console.log(`📈 [CACHE] Data extension detected: +${extInfo.new_rows_count} new rows from ${extInfo.old_end_date} to ${extInfo.new_end_date}`);
      }
    } else {
      console.log('📝 [CACHE] New data file - cache will be created after predictions');
    }
    
    // 🎯 50% 기준점 정보 로깅
    if (data.prediction_threshold) {
      console.log(`📊 [DATA ANALYSIS] Prediction threshold: ${data.prediction_threshold}`);
      console.log(`📍 [DATA ANALYSIS] 50% point: ${data.halfway_point} (${data.halfway_semimonthly})`);
      console.log(`🎯 [DATA ANALYSIS] Target period: ${data.target_semimonthly}`);
    }
    
    // 예측 가능한 시작일 목록 생성
    console.log(`📋 [DATE_PROCESSING] Raw dates from backend:`, data.dates);
    console.log(`📋 [DATE_PROCESSING] First 5 dates:`, data.dates?.slice(0, 5));
    console.log(`📋 [DATE_PROCESSING] Last 5 dates:`, data.dates?.slice(-5));
    console.log(`📋 [DATE_PROCESSING] Total dates count:`, data.dates?.length);
    
    // 26일이 있는지 특별히 확인
    if (data.dates && Array.isArray(data.dates)) {
      const has26th = data.dates.some(date => date.includes('-26'));
      const has25th = data.dates.some(date => date.includes('-25'));
      const has27th = data.dates.some(date => date.includes('-27'));
      console.log(`🔍 [DATE_CHECK] Has 25th: ${has25th}, Has 26th: ${has26th}, Has 27th: ${has27th}`);
      
      if (has26th) {
        const date26 = data.dates.find(date => date.includes('-26'));
        console.log(`📅 [DATE_CHECK] Found 26th date: ${date26}`);
        
        // 26일이 휴일인지 확인
        const is26Holiday = holidays.some(h => (h.date || h) === date26);
        console.log(`🏖️ [DATE_CHECK] Is 26th a holiday: ${is26Holiday}`);
      }
    }
    
    const startDates = generatePredictableStartDates(data.dates, holidays);
    console.log(`📋 [DATE_PROCESSING] Generated start dates:`, startDates.length);
    console.log(`📋 [DATE_PROCESSING] Holidays applied:`, holidays.length);
    console.log(`📋 [DATE_PROCESSING] First 3 start dates:`, startDates.slice(0, 3));
    console.log(`📋 [DATE_PROCESSING] Last 3 start dates:`, startDates.slice(-3));
    
    setPredictableStartDates(startDates);
    
    // 기본 선택: 가장 최근 예측 가능한 날짜 선택
    if (startDates.length > 0) {
      // startDates를 정렬해서 가장 최근 예측 시작일 찾기
      const sortedStartDates = [...startDates].sort((a, b) => b.startDate.localeCompare(a.startDate));
      const latestStartDate = sortedStartDates[0];  // 가장 최근 예측 시작일
      
      setSelectedStartDate(latestStartDate.startDate);
      setEndStartDate(latestStartDate.startDate);
      
      console.log(`🎯 [DEFAULT_SELECTION] Setting default dates:`);
      console.log(`  - Total start dates: ${startDates.length}`);
      console.log(`  - Latest start date: ${latestStartDate.startDate}`);
      console.log(`  - Required data date: ${latestStartDate.requiredDataDate}`);
      console.log(`  - All start dates:`, startDates.map(item => `${item.startDate} (data: ${item.requiredDataDate})`));
    }
    
    setIsCSVUploaded(true);
    setError(null);
    
    // 🏖️ 파일 업로드 후 휴일 정보 재로드 (데이터 빈 날짜 감지 반영)
    const reloadHolidays = async () => {
      try {
        const result = await getHolidays();
        if (result.success && result.holidays) {
          setHolidays(result.holidays);
          console.log('🏖️ [HOLIDAYS] Reloaded after file upload:', result.holidays.length);
          console.log('📊 [HOLIDAYS] File holidays:', result.file_holidays);
          console.log('🔍 [HOLIDAYS] Auto-detected holidays:', result.auto_detected_holidays);
          
          // 예측 가능한 날짜 다시 계산 (업데이트된 휴일 반영)
          if (data.dates && data.dates.length > 0) {
            const updatedStartDates = generatePredictableStartDates(data.dates, result.holidays);
            setPredictableStartDates(updatedStartDates);
            console.log('🔄 [HOLIDAYS] Updated predictable dates with new holidays:', updatedStartDates.length);
          }
        }
      } catch (error) {
        console.error('❌ [HOLIDAYS] Failed to reload holidays after file upload:', error);
      }
    };
    
    reloadHolidays();
  };

  // 단일 예측 시작
  const handleStartPrediction = async () => {
    console.log('🚀 [START] Starting single prediction...');
    
    if (!fileInfo || !fileInfo.filepath) {
      setError('파일을 먼저 업로드해주세요.');
      return;
    }

    if (!selectedStartDate || !requiredDataDate) {
      setError('예측 시작일을 선택해주세요.');
      return;
    }

    console.log('📋 [START] Prediction params:', {
      filepath: fileInfo.filepath,
      selectedStartDate: selectedStartDate,
      requiredDataDate: requiredDataDate // 백엔드에는 이 값을 전달
    });

    // 상태 초기화
    setError(null);
    setIsPredicting(true);
    setProgress(0);
    setPredictionData([]);
    setIntervalScores([]);
    setMaResults(null);
    setAttentionImage(null);

    try {
      // 백엔드에는 필요한 데이터 기준일을 전달
      const result = await startPrediction(fileInfo.filepath, requiredDataDate);
      console.log('✅ [START] Prediction started:', result);
      
      if (result.error) {
        setError(result.error);
        setIsPredicting(false);
        return;
      }
      
      checkPredictionStatus('single');
    } catch (err) {
      console.error('💥 [START] Start prediction error:', err);
      setError(err.error || '예측 시작 중 오류가 발생했습니다.');
      setIsPredicting(false);
    }
  };

  // 누적 예측 시작
  const handleStartAccumulatedPrediction = async () => {
    if (!fileInfo || !fileInfo.filepath) {
      setError('파일을 먼저 업로드해주세요.');
      return;
    }

    if (!selectedStartDate || !endStartDate) {
      setError('시작일과 종료일을 모두 선택해주세요.');
      return;
    }

    // 선택된 예측 시작일들에 해당하는 필요한 데이터 기준일 범위 계산
    const startRequiredDate = predictableStartDates.find(p => p.startDate === selectedStartDate)?.requiredDataDate;
    const endRequiredDate = predictableStartDates.find(p => p.startDate === endStartDate)?.requiredDataDate;

    if (!startRequiredDate || !endRequiredDate) {
      setError('선택된 날짜에 대한 데이터 기준일을 찾을 수 없습니다.');
      return;
    }

    setError(null);
    setIsPredicting(true);
    setProgress(0);
    
    console.log("Starting accumulated prediction:", {
      filepath: fileInfo.filepath,
      selectedStartDate: selectedStartDate,
      endStartDate: endStartDate,
      startRequiredDate: startRequiredDate,
      endRequiredDate: endRequiredDate
    });

    try {
      // 백엔드에는 필요한 데이터 기준일 범위를 전달
      await startAccumulatedPrediction(fileInfo.filepath, startRequiredDate, endRequiredDate);
      checkPredictionStatus('accumulated');
    } catch (err) {
      setError(err.error || '누적 예측 시작 중 오류가 발생했습니다.');
      setIsPredicting(false);
    }
  };

  // 예측 상태 확인
  const checkPredictionStatus = (mode = 'single') => {
    console.log(`🔄 [CHECK] Starting status check (mode: ${mode})`);
    let checkCount = 0;
    
    const statusInterval = setInterval(async () => {
      checkCount++;
      console.log(`📊 [CHECK] Status check #${checkCount}`);
      
      try {
        const status = await getPredictionStatus();
        
        console.log(`📊 [CHECK] Status received:`, status);
        setProgress(status.progress || 0);
        
        if (!status.is_predicting) {
          console.log('✅ [CHECK] Prediction completed, stopping interval');
          clearInterval(statusInterval);
          setIsPredicting(false);
          
          if (status.error) {
            console.error('❌ [CHECK] Prediction error:', status.error);
            setError(`예측 오류: ${status.error}`);
          } else {
            console.log(`🎯 [CHECK] Success, fetching results (mode: ${mode})`);
            if (mode === 'accumulated') {
              fetchAccumulatedResults();
            } else {
              setTimeout(() => {
                fetchResults();
              }, 500);
            }
          }
        }
      } catch (err) {
        console.error('💥 [CHECK] Status check error:', err);
        clearInterval(statusInterval);
        setIsPredicting(false);
        setError('예측 상태 확인 중 오류가 발생했습니다.');
      }
    }, 1000);
  };

  // 예측 결과 가져오기
  const fetchResults = async () => {
    console.log('🔄 [FETCH] Starting fetchResults...');
    setIsLoading(true);
    setError(null);
    
    try {
      const results = await getPredictionResults();
      console.log('📦 [FETCH] Raw results received:', results);
      
      if (!results || !results.success) {
        throw new Error(results?.error || '예측 결과가 없습니다');
      }
      
      console.log('📝 [STATE] Updating states:', {
        predictions: results.predictions ? results.predictions.length : 0,
        interval_scores: results.interval_scores ? results.interval_scores.length : 0,
        ma_results: !!results.ma_results,
        attention_image: !!(results.attention_data && results.attention_data.image),
        current_date: results.current_date
      });
      
      setPredictionData([...results.predictions] || []);
      setIntervalScores([...results.interval_scores] || []);
      setMaResults(results.ma_results ? {...results.ma_results} : null);
      setCurrentDate(results.current_date || null);
      
      // ✅ Attention Map 자동 로딩 - 항상 별도 API 우선 호출
      console.log('🔄 [ATTENTION_AUTO] Auto-loading attention map...');
      try {
        const attentionResult = await getAttentionMap();
        if (attentionResult.success && attentionResult.attention_data && attentionResult.attention_data.image) {
          console.log('✅ [ATTENTION_AUTO] Successfully loaded attention map from API');
          setAttentionImage(attentionResult.attention_data.image);
        } else {
          console.log('⚠️ [ATTENTION_AUTO] No attention data from API, checking main results...');
          // 백업: 메인 결과에서 확인
          if (results.attention_data && results.attention_data.image) {
            console.log('✅ [ATTENTION_AUTO] Found attention data in main results');
            setAttentionImage(results.attention_data.image);
          } else {
            console.log('ℹ️ [ATTENTION_AUTO] No attention data available anywhere');
            setAttentionImage(null);
          }
        }
      } catch (attErr) {
        console.log('⚠️ [ATTENTION_AUTO] Failed to load attention map:', attErr.message);
        // 백업: 메인 결과에서 확인
        if (results.attention_data && results.attention_data.image) {
          console.log('✅ [ATTENTION_AUTO] Using attention data from main results as fallback');
          setAttentionImage(results.attention_data.image);
        } else {
          setAttentionImage(null);
        }
      }
      
      console.log('✅ [STATE] States updated successfully');
      setActiveTab('single');
      
      // ✅ 단일 예측 완료 후 누적 예측에서도 해당 날짜 확인
      console.log('🔄 [SINGLE_TO_ACCUMULATED] Checking if this prediction can be shown in accumulated view...');
      await checkSinglePredictionInAccumulated(results.current_date);
      
    } catch (err) {
      console.error('💥 [FETCH] Catch block error:', err);
      setError(`결과 로드 오류: ${err.message || '알 수 없는 오류'}`);
    } finally {
      setIsLoading(false);
      console.log('🏁 [FETCH] fetchResults completed');
    }
  };

  // 누적 예측 결과 가져오기
  const fetchAccumulatedResults = async () => {
    console.log('🔄 [ACCUMULATED] Starting fetchAccumulatedResults...');
    setIsLoading(true);
    
    try {
      const results = await getAccumulatedResults();
      console.log('📦 [ACCUMULATED] Raw results received:', results);
      
      if (results.success) {
        console.log('✅ [ACCUMULATED] Processing successful response...');
        console.log('📊 [ACCUMULATED] Data details:', {
          predictions_length: Array.isArray(results.predictions) ? results.predictions.length : 'not array',
          accumulated_metrics: !!results.accumulated_metrics,
          accumulated_consistency_scores: !!results.accumulated_consistency_scores,
          accumulated_purchase_reliability: results.accumulated_purchase_reliability,
          accumulated_interval_scores: results.accumulated_interval_scores?.length || 'none'
        });
        
        // 데이터 안전성 검증
        const safeResults = {
          ...results,
          predictions: Array.isArray(results.predictions) ? results.predictions : [],
          accumulated_metrics: results.accumulated_metrics || {},
          accumulated_consistency_scores: results.accumulated_consistency_scores || {},
          accumulated_purchase_reliability: results.accumulated_purchase_reliability || 0
        };
        
        console.log('📝 [ACCUMULATED] Safe results prepared:', {
          predictions_count: safeResults.predictions.length,
          has_accumulated_metrics: Object.keys(safeResults.accumulated_metrics).length > 0,
          has_consistency_scores: Object.keys(safeResults.accumulated_consistency_scores).length > 0,
          purchase_reliability: safeResults.accumulated_purchase_reliability
        });
        
        setAccumulatedResults(safeResults);
        setConsistencyScores(safeResults.accumulated_consistency_scores);
        
        // ✅ 구매 신뢰도 로깅
        console.log(`💰 [ACCUMULATED] Purchase reliability received: ${safeResults.accumulated_purchase_reliability}%`);
        console.log(`🔍 [ACCUMULATED] Raw API response purchase reliability:`, results.accumulated_purchase_reliability);
        console.log(`🔍 [ACCUMULATED] Type of purchase reliability:`, typeof results.accumulated_purchase_reliability);
        console.log(`🔍 [ACCUMULATED] Full raw results object:`, JSON.stringify(results, null, 2));
        
        if (safeResults.accumulated_purchase_reliability === 100) {
          console.warn('⚠️ [ACCUMULATED] Purchase reliability is 100% - this may indicate a calculation issue');
          console.warn('⚠️ [ACCUMULATED] Debugging info:');
          console.warn('   - Raw value:', results.accumulated_purchase_reliability);
          console.warn('   - Processed value:', safeResults.accumulated_purchase_reliability);
          console.warn('   - Predictions count:', safeResults.predictions?.length || 0);
          console.warn('   - Sample prediction:', safeResults.predictions?.[0]);
          
          // ✅ 사용자에게 알림 표시
          alert(`⚠️ 구매 신뢰도가 100%로 계산되었습니다.\n\n이는 다음 중 하나일 수 있습니다:\n1. 실제로 모든 예측이 최고 점수(3점)를 받은 경우\n2. 캐시된 잘못된 데이터\n3. 계산 오류\n\n해결 방법:\n- 페이지 하단의 "누적 캐시 클리어" 버튼을 클릭\n- 다시 누적 예측 실행\n- 개발자 도구 콘솔에서 상세 로그 확인`);
        }
        
        // ✅ 캐시 통계 로깅
        if (safeResults.cache_statistics) {
          const cacheStats = safeResults.cache_statistics;
          console.log(`🎯 [CACHE] Final statistics: ${cacheStats.cached_dates}/${cacheStats.total_dates} cached (${cacheStats.cache_hit_rate?.toFixed(1)}%), ${cacheStats.new_predictions} new predictions computed`);
        }
        
        if (safeResults.predictions.length > 0) {
          const latestPrediction = safeResults.predictions[safeResults.predictions.length - 1];
          console.log('📅 [ACCUMULATED] Latest prediction:', latestPrediction);
          if (latestPrediction && latestPrediction.date) {
            setSelectedAccumulatedDate(latestPrediction.date);
            loadSelectedDatePrediction(latestPrediction.date);
          }
        }
        
        setActiveTab('accumulated');
        console.log('✅ [ACCUMULATED] Results processed successfully');
      } else {
        console.error('❌ [ACCUMULATED] API returned unsuccessful response:', results);
        setError(results.error || '누적 예측 결과가 없습니다.');
      }
    } catch (err) {
      console.error('💥 [ACCUMULATED] Catch block error:', err);
      setError(`누적 결과를 가져오는 중 오류가 발생했습니다: ${err.message || '알 수 없는 오류'}`);
    } finally {
      setIsLoading(false);
      console.log('🏁 [ACCUMULATED] fetchAccumulatedResults completed');
    }
  };

  // 단일 예측 결과를 누적 예측에서도 확인할 수 있는지 체크
  const checkSinglePredictionInAccumulated = async (currentDate) => {
    try {
      if (!currentDate) return;
      
      console.log(`🔍 [SINGLE_TO_ACCUMULATED] Checking accumulated view for date: ${currentDate}`);
      
      // 최근 누적 결과가 있는지 확인
      const recentResults = await getRecentAccumulatedResults();
      
      if (recentResults.success && recentResults.has_recent_results) {
        // 현재 단일 예측 날짜가 누적 결과에 포함되어 있는지 확인
        const isIncluded = recentResults.predictions.some(pred => pred.date === currentDate);
        
        if (isIncluded) {
          console.log(`✅ [SINGLE_TO_ACCUMULATED] Single prediction date ${currentDate} found in accumulated results`);
          
          // 누적 결과 업데이트 (이미 있는 경우)
          if (accumulatedResults) {
            console.log(`🔄 [SINGLE_TO_ACCUMULATED] Refreshing accumulated results to include latest prediction`);
            setAccumulatedResults(recentResults);
            setConsistencyScores(recentResults.accumulated_consistency_scores);
          } else {
            console.log(`📝 [SINGLE_TO_ACCUMULATED] Setting initial accumulated results`);
            setAccumulatedResults(recentResults);
            setConsistencyScores(recentResults.accumulated_consistency_scores);
          }
        } else {
          console.log(`ℹ️ [SINGLE_TO_ACCUMULATED] Single prediction date ${currentDate} not in current accumulated range`);
        }
      } else {
        console.log(`ℹ️ [SINGLE_TO_ACCUMULATED] No recent accumulated results to update`);
      }
    } catch (err) {
      console.log(`⚠️ [SINGLE_TO_ACCUMULATED] Error checking accumulated view: ${err.message}`);
      // 에러가 발생해도 단일 예측 결과에는 영향 없음
    }
  };

  // 특정 날짜의 예측 결과 로드
  const loadSelectedDatePrediction = async (date) => {
    if (!date) {
      console.warn('⚠️ [LOAD_DATE] No date provided');
      return;
    }
    
    console.log(`🔍 [LOAD_DATE] Loading prediction for date: ${date}`);
    setIsLoading(true);
    
    try {
      const result = await getAccumulatedResultByDate(date);
      console.log(`📦 [LOAD_DATE] API result for ${date}:`, result);
      
      if (result.success) {
        console.log(`✅ [LOAD_DATE] Successfully loaded data for ${date}:`, {
          predictions_count: result.predictions ? result.predictions.length : 0,
          interval_scores_count: result.interval_scores ? 
            (Array.isArray(result.interval_scores) ? result.interval_scores.length : Object.keys(result.interval_scores).length) : 0,
          metrics: result.metrics
        });
        
        // 🔍 상세 데이터 구조 확인
        if (result.predictions && result.predictions.length > 0) {
          console.log(`📊 [LOAD_DATE] First prediction sample:`, result.predictions[0]);
          console.log(`📊 [LOAD_DATE] Prediction data keys:`, Object.keys(result.predictions[0]));
        } else {
          console.warn(`⚠️ [LOAD_DATE] predictions 데이터가 비어있습니다!`);
        }
        
        if (result.interval_scores) {
          console.log(`📊 [LOAD_DATE] interval_scores 구조:`, result.interval_scores);
          if (typeof result.interval_scores === 'object' && !Array.isArray(result.interval_scores)) {
            const keys = Object.keys(result.interval_scores);
            console.log(`📊 [LOAD_DATE] interval_scores keys:`, keys);
            if (keys.length > 0) {
              console.log(`📊 [LOAD_DATE] First interval_score sample:`, result.interval_scores[keys[0]]);
            }
          }
        }
        
        // 🔧 데이터 구조 변환: 백엔드 형태 → PredictionChart 형태
        const transformedPredictions = (result.predictions || []).map((item, index) => {
          // ✅ 원본 데이터 구조 확인을 위한 상세 로깅
          if (index === 0) {
            console.log(`🔍 [LOAD_DATE] First prediction item structure:`, item);
            console.log(`🔍 [LOAD_DATE] Available keys in first item:`, Object.keys(item));
            console.log(`🔍 [LOAD_DATE] Type of item:`, typeof item);
          }
          
          // ✅ 문자열로 직렬화된 딕셔너리인 경우 파싱 처리
          let actualItem = item;
          if (typeof item === 'string' && item.startsWith('{') && item.endsWith('}')) {
            try {
              // eval 대신 안전한 방법으로 파싱 시도
              const cleanedString = item
                .replace(/Timestamp\('[^']*'\)/g, match => `"${match.slice(11, -2)}"`) // Timestamp 객체 처리
                .replace(/'/g, '"') // 작은따옴표를 큰따옴표로 변경
                .replace(/None/g, 'null'); // Python None을 JSON null로 변경
              actualItem = JSON.parse(cleanedString);
              
              if (index === 0) {
                console.log(`🔄 [LOAD_DATE] Parsed string to object:`, actualItem);
              }
            } catch (parseError) {
              console.warn(`⚠️ [LOAD_DATE] Failed to parse prediction string at index ${index}:`, parseError);
              // 파싱 실패 시 원본 사용
              actualItem = item;
            }
          }
          
          // ✅ 여러 가능한 필드명들을 확인하여 안전하게 변환
          const dateValue = actualItem.Date || actualItem.date || actualItem.prediction_date;
          const predictionValue = actualItem.Prediction || actualItem.prediction || actualItem.predicted_value || actualItem.value;
          const actualValue = actualItem.Actual || actualItem.actual || actualItem.actual_value;
          
          // ✅ 숫자 값 안전 변환
          const safePrediction = predictionValue !== null && predictionValue !== undefined ? 
            (typeof predictionValue === 'number' ? predictionValue : parseFloat(predictionValue)) : 0;
          const safeActual = actualValue !== null && actualValue !== undefined && actualValue !== 'None' ? 
            (typeof actualValue === 'number' ? actualValue : parseFloat(actualValue)) : null;
          
          // ✅ 각 필드별 상세 매핑 로깅 (첫 번째 아이템만)
          if (index === 0) {
            console.log(`🔍 [LOAD_DATE] Field mapping for first item:`, {
              dateValue,
              predictionValue,
              actualValue,
              safePrediction,
              safeActual,
              rawItem: actualItem
            });
          }
          
          return {
            Date: dateValue ? new Date(dateValue).toISOString().split('T')[0] : null,
            Prediction: safePrediction,
            Actual: safeActual
          };
        }).filter(item => item.Date !== null);
        
        // ✅ 변환 후 데이터 검증 및 로깅
        if (transformedPredictions.length > 0) {
          console.log(`🔧 [LOAD_DATE] First prediction after transform:`, transformedPredictions[0]);
          console.log(`🔧 [LOAD_DATE] Total transformed predictions:`, transformedPredictions.length);
        }
        
        // ✅ 첫 번째와 마지막 예측 값을 로깅하여 데이터 변화 확인
        if (transformedPredictions.length > 0) {
          console.log(`🔧 [LOAD_DATE] First prediction after transform:`, transformedPredictions[0]);
          console.log(`🔧 [LOAD_DATE] First prediction value: ${transformedPredictions[0]?.Prediction}`);
          console.log(`🔧 [LOAD_DATE] Last prediction: ${transformedPredictions[transformedPredictions.length-1]?.Prediction}`);
          
          // ✅ N/A 또는 undefined 값 체크
          const firstPred = transformedPredictions[0]?.Prediction;
          if (firstPred === undefined || firstPred === null || isNaN(firstPred)) {
            console.warn(`⚠️ [LOAD_DATE] First prediction value is invalid: ${firstPred} (type: ${typeof firstPred})`);
            console.warn(`⚠️ [LOAD_DATE] Original first item keys again:`, Object.keys(result.predictions[0] || {}));
            console.warn(`⚠️ [LOAD_DATE] Original first item values:`, result.predictions[0]);
          }
        }
        
        console.log(`🔧 [LOAD_DATE] Transformed data sample:`, transformedPredictions[0]);
        console.log(`🔧 [LOAD_DATE] Total transformed predictions:`, transformedPredictions.length);
        
        // ✅ 선택된 날짜의 예측 결과와 구간 점수를 별도 상태에 저장
        setSelectedDatePredictions(transformedPredictions);
        
        // ✅ interval_scores 데이터 변환 및 유효성 검사
        let intervalScoresArray = [];
        if (result.interval_scores) {
          if (Array.isArray(result.interval_scores)) {
            intervalScoresArray = result.interval_scores.filter(item => 
              item && typeof item === 'object' && 'days' in item && item.days !== null
            );
          } else if (typeof result.interval_scores === 'object') {
            intervalScoresArray = Object.values(result.interval_scores).filter(item => 
              item && typeof item === 'object' && 'days' in item && item.days !== null
            );
          }
        }
        
        console.log(`💰 [LOAD_DATE] Processed interval scores for ${date}:`, intervalScoresArray.length);
        if (intervalScoresArray.length > 0) {
          console.log(`💰 [LOAD_DATE] First interval score sample:`, intervalScoresArray[0]);
          console.log(`💰 [LOAD_DATE] Sample keys:`, Object.keys(intervalScoresArray[0]));
        }
        setSelectedDateIntervalScores(intervalScoresArray);
        
        // ✅ 일반 상태도 업데이트 (호환성을 위해)
        setPredictionData(transformedPredictions);
        setIntervalScores(intervalScoresArray);
        setCurrentDate(result.date || date);
        setMaResults(null);
        setAttentionImage(null);
        
        console.log(`🎯 [LOAD_DATE] Updated both selected and general states for ${date}`);
        console.log(`🎯 [LOAD_DATE] Final state: selectedDatePredictions=${transformedPredictions.length}, selectedDateIntervalScores=${intervalScoresArray.length}`);
      } else {
        console.error(`❌ [LOAD_DATE] Failed to load data for ${date}:`, result.error);
        setError(`${date} 날짜의 결과를 로드할 수 없습니다: ${result.error || '알 수 없는 오류'}`);
      }
    } catch (err) {
      console.error(`💥 [LOAD_DATE] Exception loading data for ${date}:`, err);
      setError(`날짜 데이터 로드 오류: ${err.message || '알 수 없는 오류'}`);
    } finally {
      setIsLoading(false);
      console.log(`🏁 [LOAD_DATE] Loading completed for ${date}`);
    }
  };

  // 보고서 다운로드
  const handleDownloadReport = () => {
    const reportUrl = getAccumulatedReportURL();
    window.open(reportUrl, '_blank');
  };

  // 새로고침 처리
  const handleRefresh = () => {
    console.log('🔄 [REFRESH] Manual refresh triggered');
    
    if (fileInfo && fileInfo.filepath) {
      if (activeTab === 'accumulated') {
        console.log('🔄 [REFRESH] Starting accumulated prediction refresh');
        handleStartAccumulatedPrediction();
      } else {
        console.log('🔄 [REFRESH] Starting single prediction refresh');
        handleStartPrediction();
      }
    } else {
      console.warn('⚠️ [REFRESH] No file info available for refresh');
      setError('새로고침하려면 파일을 먼저 업로드해주세요.');
    }
  };

  // ✅ Attention Map 자동 로딩 함수 (페이지 로드시 사용)
  const loadAttentionMapAuto = async () => {
    console.log('🔄 [ATTENTION_AUTO_LOAD] Auto-loading attention map on page load...');
    try {
      const attentionResult = await getAttentionMap();
      if (attentionResult.success && attentionResult.attention_data && attentionResult.attention_data.image) {
        console.log('✅ [ATTENTION_AUTO_LOAD] Successfully loaded attention map');
        setAttentionImage(attentionResult.attention_data.image);
        return true;
      } else {
        console.log('ℹ️ [ATTENTION_AUTO_LOAD] No attention data available');
        return false;
      }
    } catch (err) {
      console.log('⚠️ [ATTENTION_AUTO_LOAD] Error loading attention map:', err.message);
      return false;
    }
  };

  // ✅ Attention Map 수동 새로고침 함수 (버튼 클릭시 사용)
  const handleRefreshAttentionMap = async () => {
    console.log('🔄 [ATTENTION_REFRESH] Manually refreshing attention map...');
    try {
      const attentionResult = await getAttentionMap();
      if (attentionResult.success && attentionResult.attention_data && attentionResult.attention_data.image) {
        console.log('✅ [ATTENTION_REFRESH] Successfully refreshed attention map');
        setAttentionImage(attentionResult.attention_data.image);
      } else {
        console.log('⚠️ [ATTENTION_REFRESH] No attention data available');
        alert('현재 사용 가능한 Attention Map 데이터가 없습니다.');
        setAttentionImage(null);
      }
    } catch (err) {
      console.error('💥 [ATTENTION_REFRESH] Error refreshing attention map:', err);
      alert(`Attention Map 새로고침 중 오류: ${err.message}`);
    }
  };

  // 누적 예측 날짜 선택 시
  const handleAccumulatedDateSelect = (date) => {
    console.log(`🎯 [SELECT] Date selected: ${date}`);
    console.log(`🎯 [SELECT] Previous selected date: ${selectedAccumulatedDate}`);
    
    // ✅ 선택된 날짜 상태를 먼저 업데이트
    setSelectedAccumulatedDate(date);
    
    // ✅ 기존 상태 초기화
    setSelectedDatePredictions([]);
    setSelectedDateIntervalScores([]);
    
    console.log(`🎯 [SELECT] Loading prediction data for ${date}...`);
    loadSelectedDatePrediction(date);
  };

  // 누적 예측에서 단일 예측으로 전환
  const handleViewInSinglePrediction = async (date) => {
    try {
      console.log(`🔄 [ACCUMULATED_TO_SINGLE] Switching to single prediction view for date: ${date}`);
      setIsLoading(true);
      
      // 해당 날짜의 상세 예측 결과 로드
      const result = await getAccumulatedResultByDate(date);
      
      if (result.success) {
        // 단일 예측 형태로 데이터 변환
        const transformedPredictions = (result.predictions || []).map(item => ({
          Date: item.date || item.Date,
          Prediction: item.prediction || item.Prediction,
          Actual: item.actual || item.Actual || null
        }));

        // 구간 점수 변환
        let intervalScoresArray = [];
        if (result.interval_scores) {
          if (Array.isArray(result.interval_scores)) {
            intervalScoresArray = result.interval_scores.filter(item => 
              item && typeof item === 'object' && 'days' in item && item.days !== null
            );
          } else if (typeof result.interval_scores === 'object') {
            intervalScoresArray = Object.values(result.interval_scores).filter(item => 
              item && typeof item === 'object' && 'days' in item && item.days !== null
            );
          }
        }

        // 단일 예측 상태로 설정
        setPredictionData(transformedPredictions);
        setIntervalScores(intervalScoresArray);
        setCurrentDate(date);
        
        // 🎯 Attention 데이터 설정 (디버깅 로그 추가)
        console.log(`🔍 [ACCUMULATED_TO_SINGLE] Raw attention_data:`, result.attention_data);
        if (result.attention_data) {
          console.log(`🔍 [ACCUMULATED_TO_SINGLE] Attention data keys:`, Object.keys(result.attention_data));
          if (result.attention_data.image_base64) {
            setAttentionImage(result.attention_data.image_base64);
            console.log(`✅ [ACCUMULATED_TO_SINGLE] Attention image set from image_base64`);
          } else if (result.attention_data.image) {
            setAttentionImage(result.attention_data.image);
            console.log(`✅ [ACCUMULATED_TO_SINGLE] Attention image set from image`);
          } else {
            setAttentionImage(null);
            console.log(`⚠️ [ACCUMULATED_TO_SINGLE] No attention image in data structure`);
          }
        } else {
          setAttentionImage(null);
          console.log(`⚠️ [ACCUMULATED_TO_SINGLE] No attention_data available`);
        }

        // 📊 이동평균 결과 설정 (디버깅 로그 추가)
        console.log(`🔍 [ACCUMULATED_TO_SINGLE] Raw ma_results:`, result.ma_results);
        if (result.ma_results && Object.keys(result.ma_results).length > 0) {
          setMaResults(result.ma_results);
          console.log(`✅ [ACCUMULATED_TO_SINGLE] MA results loaded: ${Object.keys(result.ma_results).length} windows`);
          console.log(`🔍 [ACCUMULATED_TO_SINGLE] MA windows:`, Object.keys(result.ma_results));
        } else {
          setMaResults(null);
          console.log(`⚠️ [ACCUMULATED_TO_SINGLE] No MA results available for ${date}`);
        }

        // 단일 예측 탭으로 전환
        setActiveTab('single');
        
        console.log(`✅ [ACCUMULATED_TO_SINGLE] Successfully switched to single view for ${date}`);
        console.log(`📊 [ACCUMULATED_TO_SINGLE] Data loaded: ${transformedPredictions.length} predictions, ${intervalScoresArray.length} intervals`);
        
      } else {
        console.error(`❌ [ACCUMULATED_TO_SINGLE] Failed to load data for ${date}:`, result.error);
        setError(`선택한 날짜의 상세 데이터를 불러올 수 없습니다: ${result.error}`);
      }
    } catch (err) {
      console.error(`💥 [ACCUMULATED_TO_SINGLE] Error switching to single view: ${err.message}`);
      setError(`단일 예측 전환 중 오류가 발생했습니다: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // 누적 예측 캐시 클리어
  const handleClearAccumulatedCache = async () => {
    try {
      const result = await clearAccumulatedCache();
      if (result.success) {
        console.log('✅ [CACHE] Cache cleared successfully');
        // 상태 초기화
        setAccumulatedResults(null);
        setConsistencyScores(null);
        setSelectedAccumulatedDate(null);
        setPredictionData([]);
        setIntervalScores([]);
        setSelectedDatePredictions([]);
        setSelectedDateIntervalScores([]);
        alert('누적 예측 캐시가 클리어되었습니다. 다시 누적 예측을 실행해주세요.');
      } else {
        console.error('❌ [CACHE] Cache clear failed:', result.error);
        alert('캐시 클리어에 실패했습니다: ' + result.error);
      }
    } catch (err) {
      console.error('💥 [CACHE] Cache clear error:', err);
      alert('캐시 클리어 중 오류가 발생했습니다.');
    }
  };

  // 앱 전체에서 사용할 스타일 정의 (수정됨)
  const styles = {
    appContainer: {
      display: 'flex',
      flexDirection: 'column',
      minHeight: '100vh',
      backgroundColor: '#ffffff'
    },
    header: {
      backgroundColor: '#ffffff',
      color: '#374151',
      padding: '1rem 1.5rem',
      boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
      borderBottom: '1px solid #e5e7eb'
    },
    headerContent: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      flexWrap: 'wrap',
      gap: '1rem'
    },
    headerTitle: {
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      marginLeft: '2rem'
    },
    headerTabs: {
      display: 'flex',
      alignItems: 'center',
      gap: '2rem'
    },
    headerTab: (isActive) => ({
      padding: '0.5rem 1rem',
      cursor: 'pointer',
      fontWeight: isActive ? '600' : '500',
      fontSize: '1rem',
      color: isActive ? '#2563eb' : '#6b7280',
      borderBottom: isActive ? '2px solid #2563eb' : '2px solid transparent',
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      transition: 'all 0.2s',
      borderRadius: '0.375rem 0.375rem 0 0',
      position: 'relative'
    }),
    dropdown: {
      position: 'absolute',
      top: '100%',
      left: '0',
      backgroundColor: '#ffffff',
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
      borderRadius: '0 0 0.5rem 0.5rem',
      border: '1px solid #e5e7eb',
      borderTop: 'none',
      minWidth: '200px',
      opacity: '0',
      visibility: 'hidden',
      transform: 'translateY(-10px)',
      transition: 'all 0.2s ease',
      zIndex: 1000
    },
    dropdownItem: {
      padding: '0.75rem 1rem',
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem',
      cursor: 'pointer',
      fontSize: '0.875rem',
      color: '#374151',
      borderBottom: '1px solid #f3f4f6',
      transition: 'background-color 0.2s',
      '&:hover': {
        backgroundColor: '#f9fafb'
      },
      '&:last-child': {
        borderBottom: 'none'
      }
    },
    headerInfo: {
      display: 'flex',
      alignItems: 'center',
      gap: '1rem',
      color: '#6b7280',
      fontSize: '0.875rem'
    },
    titleText: {
      fontSize: '1.5rem',
      fontWeight: '700',
      color: '#1f2937'
    },
    subTabContainer: {
      backgroundColor: '#ffffff',
      borderRadius: '0.5rem 0.5rem 0 0',
      boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
      marginBottom: '0',
      overflow: 'hidden'
    },
    subTab: (isActive) => ({
      display: 'inline-flex',
      alignItems: 'center',
      gap: '0.5rem',
      padding: '0.875rem 1.5rem',
      cursor: 'pointer',
      fontWeight: isActive ? '600' : '500',
      fontSize: '0.9rem',
      color: isActive ? '#ffffff' : '#6b7280',
      backgroundColor: isActive ? '#1e40af' : '#ffffff',
      borderRight: '1px solid #e5e7eb',
      transition: 'all 0.2s ease',
      '&:last-child': {
        borderRight: 'none'
      }
    }),
    mainContent: {
      flex: 1,
      backgroundColor: '#f3f4f6',
      paddingTop: '2rem',
      paddingBottom: '2rem',
      paddingLeft: '1.5rem',
      paddingRight: '1.5rem'
    },
    card: {
      backgroundColor: 'white',
      borderRadius: '0 0 0.5rem 0.5rem',
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
      padding: '1rem',
      marginBottom: '1.5rem'
    },
    cardHeader: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      marginBottom: '1rem'
    },
    cardTitle: {
      fontSize: '1.125rem',
      fontWeight: '600',
      display: 'flex',
      alignItems: 'center'
    },
    iconStyle: {
      marginRight: '0.5rem',
      color: '#2563eb'
    },
    refreshButton: {
      display: 'flex',
      alignItems: 'center',
      gap: '0.25rem',
      backgroundColor: '#dbeafe',
      color: '#1e40af',
      padding: '0.25rem 0.75rem',
      borderRadius: '0.375rem',
      cursor: 'pointer',
      border: 'none'
    },
    dateSelectContainer: {
      marginTop: '1.5rem',
      display: 'flex',
      flexDirection: 'column',
      gap: '1.5rem'
    },
    selectContainer: {
      flex: windowWidth < 768 ? '1' : '0 1 350px',
      minWidth: windowWidth < 768 ? '100%' : '280px',
      maxWidth: windowWidth < 768 ? '100%' : '350px',
      position: 'relative'
    },
    selectRow: {
      display: 'flex',
      flexDirection: windowWidth < 768 ? 'column' : 'row',
      alignItems: windowWidth < 768 ? 'stretch' : 'flex-start',
      gap: windowWidth < 768 ? '1.5rem' : '2.5rem',
      flexWrap: 'wrap',
      justifyContent: windowWidth < 768 ? 'stretch' : 'flex-start'
    },
    buttonRow: {
      display: 'flex',
      flexDirection: windowWidth < 768 ? 'column' : 'row',
      alignItems: windowWidth < 768 ? 'stretch' : 'center',
      gap: '1rem',
      marginTop: '1rem'
    },
    selectLabel: {
      display: 'block',
      fontSize: '0.875rem',
      fontWeight: '500',
      color: '#374151',
      marginBottom: '0.5rem'
    },
    calendarWrapper: {
      position: 'relative',
      zIndex: 1000,
      width: '100%',
      maxWidth: '350px'
    },
    predictionButton: {
      backgroundColor: '#10b981',
      color: 'white',
      padding: '0.875rem 1.5rem',
      borderRadius: '0.5rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '0.5rem',
      cursor: 'pointer',
      border: 'none',
      fontSize: '0.875rem',
      fontWeight: '500',
      whiteSpace: 'nowrap',
      minWidth: '180px',
      boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
      transition: 'all 0.2s'
    },
    accumulatedButton: {
      backgroundColor: '#8b5cf6',
      color: 'white',
      padding: '0.875rem 1.5rem',
      borderRadius: '0.5rem',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '0.5rem',
      cursor: 'pointer',
      border: 'none',
      fontSize: '0.875rem',
      fontWeight: '500',
      whiteSpace: 'nowrap',
      minWidth: '180px',
      boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
      transition: 'all 0.2s'
    },
    progressContainer: {
      marginTop: '1rem'
    },
    progressText: {
      fontSize: '0.875rem',
      color: '#6b7280',
      marginBottom: '0.25rem'
    },
    errorMessage: {
      color: '#ef4444',
      display: 'flex',
      alignItems: 'center',
      marginTop: '0.75rem'
    },
    dashboardGrid: {
      display: 'grid',
      gridTemplateColumns: '1fr',
      gap: '1.5rem',
      '@media (min-width: 768px)': {
        gridTemplateColumns: 'repeat(2, 1fr)'
      }
    },
    footer: {
      backgroundColor: '#f3f4f6',
      padding: '1rem',
      textAlign: 'center',
      color: '#6b7280',
      fontSize: '0.875rem'
    },
    helpText: {
      marginTop: '0.5rem',
      fontSize: '0.875rem',
      color: '#6b7280'
    },
    tabContainer: {
      display: 'flex',
      borderBottom: '1px solid #e5e7eb',
      marginBottom: '1rem'
    },
    tab: (isActive) => ({
      padding: '0.75rem 1rem',
      cursor: 'pointer',
      fontWeight: isActive ? '500' : 'normal',
      color: isActive ? '#2563eb' : '#6b7280',
      borderBottom: isActive ? '2px solid #2563eb' : 'none',
      display: 'flex',
      alignItems: 'center',
      gap: '0.5rem'
    }),
    scrollableTable: {
      maxHeight: '400px',
      overflowY: 'auto'
    },
    predictionPreview: {
      padding: '1rem',
      backgroundColor: '#f0f9ff',
      borderRadius: '0.5rem',
      border: '1px solid #bae6fd',
      boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
      maxWidth: '100%'
    },
    accumulatedPreview: {
      padding: '1rem',
      backgroundColor: '#faf5ff',
      borderRadius: '0.5rem',
      border: '1px solid #ddd6fe',
      boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)',
      maxWidth: '100%'
    },
    previewText: {
      fontSize: '0.875rem',
      fontWeight: '500',
      color: '#1e40af',
      margin: '0 0 0.25rem 0'
    },
    previewHelpText: {
      margin: '0.25rem 0 0 0',
      fontSize: '0.75rem',
      color: '#6b7280',
      lineHeight: '1.4'
    },
    exampleBox: {
      marginTop: '0.5rem',
      padding: '0.5rem',
      backgroundColor: '#f8fafc',
      borderRadius: '0.25rem',
      border: '1px solid #e2e8f0'
    },
    exampleTitle: {
      fontSize: '0.75rem',
      fontWeight: '600',
      color: '#475569',
      margin: '0 0 0.25rem 0'
    },
    exampleItem: {
      fontSize: '0.7rem',
      color: '#64748b',
      margin: '0.1rem 0',
      lineHeight: '1.3'
    },
    // 홈 페이지 스타일
    homeContainer: {
      display: 'flex',
      flexDirection: 'column',
      gap: '2rem'
    },
    welcomeCard: {
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      borderRadius: '1rem',
      padding: '3rem 2rem',
      color: 'white',
      textAlign: 'center'
    },
    welcomeContent: {
      maxWidth: '800px',
      margin: '0 auto'
    },
    welcomeTitle: {
      fontSize: windowWidth < 768 ? '1.8rem' : '2.5rem',
      fontWeight: '700',
      marginBottom: '1rem',
      lineHeight: '1.2',
      whiteSpace: windowWidth < 768 ? 'normal' : 'nowrap',
      textAlign: 'center'
    },
    welcomeSubtitle: {
      fontSize: '1.25rem',
      opacity: '0.9',
      lineHeight: '1.6'
    },
    featuresGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
      gap: '2rem'
    },
    featureCard: {
      backgroundColor: 'white',
      borderRadius: '1rem',
      padding: '2rem',
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
      textAlign: 'center',
      transition: 'transform 0.2s, box-shadow 0.2s',
      '&:hover': {
        transform: 'translateY(-4px)',
        boxShadow: '0 8px 15px -3px rgba(0, 0, 0, 0.1)'
      }
    },
    featureIcon: {
      display: 'flex',
      justifyContent: 'center',
      marginBottom: '1.5rem'
    },
    featureTitle: {
      fontSize: '1.5rem',
      fontWeight: '600',
      marginBottom: '1rem',
      color: '#1f2937'
    },
    featureDescription: {
      fontSize: '1rem',
      color: '#6b7280',
      lineHeight: '1.6',
      marginBottom: '1.5rem'
    },
    featureHighlights: {
      display: 'flex',
      flexDirection: 'column',
      gap: '0.5rem',
      alignItems: 'center'
    },
    highlight: {
      fontSize: '0.875rem',
      color: '#2563eb',
      fontWeight: '500'
    },
    modelInfoCard: {
      backgroundColor: 'white',
      borderRadius: '1rem',
      padding: '2rem',
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
    },
    modelTitle: {
      fontSize: '1.75rem',
      fontWeight: '600',
      marginBottom: '2rem',
      color: '#1f2937',
      display: 'flex',
      alignItems: 'center'
    },
    modelDetails: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
      gap: '2rem'
    },
    modelFeature: {
      padding: '1.5rem',
      backgroundColor: '#f9fafb',
      borderRadius: '0.75rem',
      borderLeft: '4px solid #2563eb'
    },
    guideCard: {
      backgroundColor: 'white',
      borderRadius: '1rem',
      padding: '2rem',
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
    },
    guideTitle: {
      fontSize: '1.75rem',
      fontWeight: '600',
      marginBottom: '2rem',
      color: '#1f2937',
      display: 'flex',
      alignItems: 'center'
    },
    guideSteps: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
      gap: '1.5rem'
    },
    guideStep: {
      display: 'flex',
      alignItems: 'flex-start',
      gap: '1rem'
    },
    stepNumber: {
      width: '2.5rem',
      height: '2.5rem',
      backgroundColor: '#2563eb',
      color: 'white',
      borderRadius: '50%',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontWeight: '600',
      fontSize: '1.125rem',
      flexShrink: 0
    },
    stepContent: {
      flex: 1
    }
  };

  return (
    <div style={styles.appContainer}>
      <style>{dropdownCSS}</style>
      {/* 헤더 */}
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <div style={styles.headerTitle}>
            <img 
              src={`${process.env.PUBLIC_URL}/Lotte_Chem_logo.png`} 
              alt="롯데케미칼 로고" 
              style={{
                height: '36px',
                width: 'auto',
                marginRight: '12px',
                cursor: 'pointer'
              }}
              onClick={() => setSystemTab('home')}
            />
            <h1 style={styles.titleText}>롯데케미칼 MOPJ 가격 예측 및 구매 전략 대시보드</h1>
          </div>
          <div style={styles.headerTabs}>
            <div 
              style={styles.headerTab(systemTab === 'home')}
              onClick={() => setSystemTab('home')}
            >
              <Grid size={16} />
              홈
            </div>
            <div 
              style={styles.headerTab(systemTab === 'prediction')}
              onClick={() => setSystemTab('prediction')}
              onMouseEnter={(e) => {
                if (systemTab !== 'prediction') {
                  e.currentTarget.querySelector('.dropdown')?.classList.add('show');
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.querySelector('.dropdown')?.classList.remove('show');
              }}
            >
              <TrendingUp size={16} />
              예측 시스템
              <div className="dropdown" style={styles.dropdown}>
                <div style={styles.dropdownItem} onClick={(e) => {
                  e.stopPropagation();
                  setSystemTab('prediction');
                  setActiveTab('single');
                }}>
                  <TrendingUp size={14} />
                  단일 날짜 예측
                </div>
                <div style={styles.dropdownItem} onClick={(e) => {
                  e.stopPropagation();
                  setSystemTab('prediction');
                  setActiveTab('accumulated');
                }}>
                  <Activity size={14} />
                  누적 예측 분석
                </div>
              </div>
            </div>
            <div 
              style={styles.headerTab(systemTab === 'settings')}
              onClick={() => setSystemTab('settings')}
            >
              <Calendar size={16} />
              휴일 설정
            </div>
          </div>
          <div style={styles.headerInfo}>
            {currentDate && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Calendar size={18} />
                <span>예측 시작일: {currentDate}</span>
              </div>
            )}
          </div>
        </div>
      </header>

      <main style={styles.mainContent}>
        {/* 예측 시스템 서브탭 */}
        {systemTab === 'prediction' && (
          <div style={styles.subTabContainer}>
            <div 
              style={styles.subTab(activeTab === 'single')}
              onClick={() => setActiveTab('single')}
            >
              <TrendingUp size={16} />
              단일 날짜 예측
            </div>
            <div 
              style={styles.subTab(activeTab === 'accumulated')}
              onClick={() => setActiveTab('accumulated')}
            >
              <Activity size={16} />
              누적 예측 분석
            </div>
          </div>
        )}

        {/* 홈 페이지 */}
        {systemTab === 'home' && (
          <div style={styles.homeContainer}>
            {/* 웰컴 섹션 */}
            <div style={styles.welcomeCard}>
              <div style={styles.welcomeContent}>
                <h2 style={styles.welcomeTitle}>MOPJ 가격 예측 시스템에 오신 것을 환영합니다</h2>
                <p style={styles.welcomeSubtitle}>
                  MOPJ(Mean Of Platts Japan) 가격을 AI 딥러닝 기술로 정확하게 예측하는 전문 시스템입니다.
                </p>
              </div>
            </div>

            {/* 주요 기능 섹션 */}
            <div style={styles.featuresGrid}>
              <div style={styles.featureCard}>
                <div style={styles.featureIcon}>
                  <TrendingUp size={32} style={{ color: '#2563eb' }} />
                </div>
                <h3 style={styles.featureTitle}>단일 날짜 예측</h3>
                <p style={styles.featureDescription}>
                  특정 날짜를 선택하여 23일간의 MOPJ 가격을 상세하게 예측합니다.
                  실시간 차트와 구간별 신뢰도를 제공합니다.
                </p>
                <div style={styles.featureHighlights}>
                  <span style={styles.highlight}>• 23일 예측</span>
                  <span style={styles.highlight}>• 실시간 차트</span>
                  <span style={styles.highlight}>• 신뢰도 분석</span>
                </div>
              </div>

              <div style={styles.featureCard}>
                <div style={styles.featureIcon}>
                  <Activity size={32} style={{ color: '#8b5cf6' }} />
                </div>
                <h3 style={styles.featureTitle}>누적 예측 분석</h3>
                <p style={styles.featureDescription}>
                  연속된 기간 동안의 예측을 수행하여 장기적인 가격 트렌드를 분석합니다.
                  누적 정확도와 일관성 점수를 제공합니다.
                </p>
                <div style={styles.featureHighlights}>
                  <span style={styles.highlight}>• 연속 예측</span>
                  <span style={styles.highlight}>• 트렌드 분석</span>
                  <span style={styles.highlight}>• 일관성 평가</span>
                </div>
              </div>

              <div style={styles.featureCard}>
                <div style={styles.featureIcon}>
                  <BarChart size={32} style={{ color: '#10b981' }} />
                </div>
                <h3 style={styles.featureTitle}>고급 분석 도구</h3>
                <p style={styles.featureDescription}>
                  이동평균, 어텐션 맵, 구간별 점수 등 다양한 분석 도구로
                  예측 결과를 다각도로 검증할 수 있습니다.
                </p>
                <div style={styles.featureHighlights}>
                  <span style={styles.highlight}>• 이동평균 분석</span>
                  <span style={styles.highlight}>• 어텐션 맵</span>
                  <span style={styles.highlight}>• 구간별 평가</span>
                </div>
              </div>
            </div>

            {/* 모델 정보 섹션 */}
            <div style={styles.modelInfoCard}>
              <h3 style={styles.modelTitle}>
                <Award size={24} style={{ color: '#f59e0b', marginRight: '0.5rem' }} />
                AI 딥러닝 예측 모델
              </h3>
              <div style={styles.modelDetails}>
                <div style={styles.modelFeature}>
                  <h4>🧠 LSTM + Attention 하이브리드 아키텍처</h4>
                  <p>계층적 LSTM과 듀얼 어텐션 메커니즘(시간적/특징 어텐션)을 결합하여 시계열 데이터의 복잡한 패턴을 학습합니다.</p>
                </div>
                <div style={styles.modelFeature}>
                  <h4>📊 다변량 시계열 분석</h4>
                  <p>MOPJ 가격뿐만 아니라 관련 경제 지표들을 종합적으로 분석하여 예측 정확도를 높입니다.</p>
                </div>
                <div style={styles.modelFeature}>
                  <h4>🎯 실시간 캐싱 시스템</h4>
                  <p>효율적인 캐싱 메커니즘으로 빠른 예측 결과를 제공하며, 기존 데이터를 활용한 증분 예측이 가능합니다.</p>
                </div>
              </div>
            </div>

            {/* 사용 방법 안내 */}
            <div style={styles.guideCard}>
              <h3 style={styles.guideTitle}>
                <Database size={24} style={{ color: '#2563eb', marginRight: '0.5rem' }} />
                사용 방법
              </h3>
              <div style={styles.guideSteps}>
                <div style={styles.guideStep}>
                  <div style={styles.stepNumber}>1</div>
                  <div style={styles.stepContent}>
                    <h4>데이터 업로드</h4>
                    <p>CSV 형식의 MOPJ 가격 데이터를 업로드합니다.</p>
                  </div>
                </div>
                <div style={styles.guideStep}>
                  <div style={styles.stepNumber}>2</div>
                  <div style={styles.stepContent}>
                    <h4>예측 모드 선택</h4>
                    <p>단일 날짜 예측 또는 누적 예측 분석 중 선택합니다.</p>
                  </div>
                </div>
                <div style={styles.guideStep}>
                  <div style={styles.stepNumber}>3</div>
                  <div style={styles.stepContent}>
                    <h4>날짜 설정 & 실행</h4>
                    <p>예측 시작일을 선택하고 예측을 시작합니다.</p>
                  </div>
                </div>
                <div style={styles.guideStep}>
                  <div style={styles.stepNumber}>4</div>
                  <div style={styles.stepContent}>
                    <h4>결과 분석</h4>
                    <p>차트, 표, 분석 도구로 예측 결과를 확인합니다.</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 휴일 관리 탭 */}
        {systemTab === 'settings' && (
          <div style={styles.card}>
            <HolidayManager />
          </div>
        )}

        {/* 예측 시스템 탭 */}
        {systemTab === 'prediction' && (
          <>
            {/* 데이터 업로드 섹션 */}
            <div style={styles.card}>
              <div style={styles.cardHeader}>
                <h2 style={styles.cardTitle}>
                  <Database size={18} style={styles.iconStyle} />
                  데이터 입력
                </h2>
                {isCSVUploaded && (
                  <button
                    style={styles.refreshButton}
                    onClick={handleRefresh}
                    disabled={isLoading || isPredicting}
                  >
                    <RefreshCw size={16} style={isLoading || isPredicting ? { animation: 'spin 1s linear infinite' } : {}} />
                    <span>새로고침</span>
                  </button>
                )}
              </div>
              

              
              {/* 파일 업로드 컴포넌트 */}
              {!isCSVUploaded && (
                <FileUploader 
                  onUploadSuccess={handleUploadSuccess}
                  isLoading={isLoading}
                  setIsLoading={setIsLoading}
                />
              )}
              
              {/* 📂 파일 업로드 완료 및 캐시 정보 표시 */}
              {isCSVUploaded && fileInfo && (
                <div style={{
                  padding: '1rem',
                  backgroundColor: '#f0f9ff',
                  borderRadius: '0.5rem',
                  border: '1px solid #bae6fd',
                  marginBottom: '1rem'
                }}>
                  <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.5rem',
                    marginBottom: '0.5rem'
                  }}>
                    <Database size={16} style={{ color: '#2563eb' }} />
                    <strong style={{ color: '#1e40af' }}>파일 업로드 완료</strong>
                  </div>
                  <div style={{ fontSize: '0.875rem', color: '#64748b', lineHeight: '1.4' }}>
                    📄 <strong>파일:</strong> {fileInfo.original_filename || fileInfo.filename}<br/>
                    📊 <strong>데이터 날짜:</strong> {fileInfo.dates && fileInfo.dates.length > 0 && `${fileInfo.dates[fileInfo.dates.length - 1]} (총 ${fileInfo.dates.length}일)`}
                  </div>
                  
                  {/* 🎯 캐시 정보 표시 */}
                  {fileInfo.cache_info && (
                    <div style={{
                      marginTop: '0.75rem',
                      padding: '0.75rem',
                      backgroundColor: fileInfo.cache_info.found ? '#f0f9ff' : '#fefce8',
                      borderRadius: '0.375rem',
                      border: `1px solid ${fileInfo.cache_info.found ? '#bae6fd' : '#fef3c7'}`
                    }}>
                      <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        marginBottom: '0.25rem'
                      }}>
                        <span style={{ fontSize: '1rem' }}>
                          {fileInfo.cache_info.found ? '⚡' : '📝'}
                        </span>
                        <strong style={{
                          color: fileInfo.cache_info.found ? '#1e40af' : '#d97706',
                          fontSize: '0.875rem'
                        }}>
                          {fileInfo.cache_info.found ? '캐시 활용 가능' : '새 데이터'}
                        </strong>
                      </div>
                      <div style={{ fontSize: '0.75rem', color: '#6b7280', lineHeight: '1.4' }}>
                        {fileInfo.cache_info.message}
                        {fileInfo.cache_info.cache_type === 'exact' && (
                          <><br/>✨ <strong>기존 예측 결과를 즉시 불러올 수 있습니다!</strong></>
                        )}
                        {fileInfo.cache_info.cache_type === 'extension' && (
                          <><br/>🚀 <strong>기존 캐시를 활용하여 새 부분만 계산합니다!</strong></>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              {/* 개선된 단일 예측 날짜 선택 - 달력 적용 */}
              {isCSVUploaded && activeTab === 'single' && (
                <div style={styles.dateSelectContainer}>
                  <div style={styles.selectRow}>
                    <div style={styles.selectContainer}>
                      <label style={styles.selectLabel}>
                        🚀 예측 시작일 선택
                      </label>
                      <div style={styles.calendarWrapper}>
                        <CalendarDatePicker
                          availableDates={predictableStartDates}
                          selectedDate={selectedStartDate}
                          onDateSelect={setSelectedStartDate}
                          title="예측 시작일을 선택하세요"
                          holidays={holidays}
                        />
                      </div>
                    </div>
                  </div>

                  {/* 필요한 데이터 기준일 미리보기 */}
                  {selectedStartDate && requiredDataDate && (
                    <div style={styles.predictionPreview}>
                      <p style={styles.previewText}>
                        🚀 <strong>예측 시작일:</strong> {formatDate(selectedStartDate)}
                      </p>
                      <p style={styles.previewText}>
                        📊 <strong>사용할 데이터:</strong> {formatDate(requiredDataDate)}까지
                      </p>
                      <p style={styles.previewText}>
                        📈 <strong>예측 기간:</strong> {formatDate(selectedStartDate)}부터 23일간
                      </p>
                      
                      <p style={styles.previewHelpText}>
                        💡 {formatDate(requiredDataDate)}까지의 데이터를 사용하여 {formatDate(selectedStartDate)}부터 23일간 예측합니다.
                      </p>
                      <p style={styles.previewHelpText}>
                        📅 달력에 표시되는 날짜는 실제 예측이 시작되는 날짜입니다.
                      </p>
                      {holidays.length > 0 && (
                        <p style={styles.previewHelpText}>
                          🏖️ 휴일은 참조용으로 표시되며, 주말이 아닌 날짜는 모두 선택 가능합니다.
                        </p>
                      )}
                  </div>
                )}
                  
                  <div style={styles.buttonRow}>
                    <button
                      style={styles.predictionButton}
                      onClick={handleStartPrediction}
                      disabled={isPredicting || !selectedStartDate}
                    >
                      <TrendingUp size={18} />
                      {isPredicting 
                        ? '예측 중...' 
                        : selectedStartDate 
                          ? `${formatDate(selectedStartDate)}부터 예측 시작`
                          : '날짜 선택 후 예측'
                      }
                    </button>
                  </div>
                </div>
              )}
              
              {/* 개선된 누적 예측 날짜 선택 - 달력 적용 */}
              {isCSVUploaded && activeTab === 'accumulated' && (
                <div style={styles.dateSelectContainer}>
                  <div style={styles.selectRow}>
                    <div style={styles.selectContainer}>
                      <label style={styles.selectLabel}>
                        🚀 누적 예측 시작일
                      </label>
                      <div style={styles.calendarWrapper}>
                        <CalendarDatePicker
                          availableDates={predictableStartDates}
                          selectedDate={selectedStartDate}
                          onDateSelect={setSelectedStartDate}
                          title="시작일을 선택하세요"
                          holidays={holidays}
                        />
                      </div>
                    </div>
                    
                    <div style={styles.selectContainer}>
                      <label style={styles.selectLabel}>
                        🏁 누적 예측 종료일
                      </label>
                      <div style={styles.calendarWrapper}>
                        <CalendarDatePicker
                          availableDates={predictableStartDates.filter(item => 
                            !selectedStartDate || item.startDate >= selectedStartDate
                          )}
                          selectedDate={endStartDate}
                          onDateSelect={setEndStartDate}
                          title="종료일을 선택하세요"
                          holidays={holidays}
                        />
                      </div>
                    </div>
                  </div>
                  
                  {/* 누적 예측 미리보기 */}
                  {selectedStartDate && endStartDate && (() => {
                    const preview = calculateAccumulatedPreview(selectedStartDate, endStartDate);
                    return preview && (
                      <div style={styles.accumulatedPreview}>
                        <p style={styles.previewText}>
                          🔄 <strong>수행할 예측 횟수:</strong> {preview.predictionCount}회
                        </p>
                        <p style={styles.previewText}>
                          📅 <strong>예측 기간:</strong> {formatDate(preview.firstPredictionStart)} ~ {formatDate(preview.lastPredictionStart)}
                        </p>
                        <div style={styles.exampleBox}>
                          <p style={styles.exampleTitle}>📋 <strong>예측 수행 예시:</strong></p>
                          {preview.predictionDates.slice(0, 3).map((item, index) => (
                            <p key={index} style={styles.exampleItem}>
                              • {formatDate(item.startDate)}부터 예측 (데이터: {formatDate(item.requiredDataDate)})
                            </p>
                          ))}
                          {preview.predictionCount > 3 && (
                            <p style={styles.exampleItem}>
                              ... 총 {preview.predictionCount}회 예측 수행
                            </p>
                          )}
                        </div>
                        <p style={styles.previewHelpText}>
                          💡 각 예측 시작일에 맞는 데이터를 사용하여 해당 날짜부터 23일간 예측을 수행합니다.
                        </p>
                        <p style={styles.previewHelpText}>
                          📅 달력에 표시되는 날짜는 실제 예측이 시작되는 날짜입니다.
                        </p>
                        {holidays.length > 0 && (
                          <p style={styles.previewHelpText}>
                            🏖️ 휴일은 참조용으로 표시되며, 주말이 아닌 날짜는 모두 선택 가능합니다.
                          </p>
                        )}
                        
                        {/* 캐시 정보 표시 */}
                        {cacheInfo && (
                          <div style={{
                            marginTop: '0.75rem',
                            padding: '0.75rem',
                            backgroundColor: cacheInfo.cache_percentage > 0 ? '#f0f9ff' : '#fef3f2',
                            borderRadius: '0.375rem',
                            border: `1px solid ${cacheInfo.cache_percentage > 0 ? '#bae6fd' : '#fecaca'}`
                          }}>
                            <div style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.5rem',
                              marginBottom: '0.5rem'
                            }}>
                              <span style={{ fontSize: '1rem' }}>
                                {cacheInfo.cache_percentage > 0 ? '⚡' : '🔄'}
                              </span>
                              <strong style={{
                                color: cacheInfo.cache_percentage > 0 ? '#1e40af' : '#dc2626',
                                fontSize: '0.875rem'
                              }}>
                                캐시 활용률: {cacheInfo.cache_percentage}%
                              </strong>
                            </div>
                            <div style={{ fontSize: '0.75rem', color: '#6b7280', lineHeight: '1.4' }}>
                              📊 저장된 예측: {cacheInfo.cached_predictions}개 / 전체: {cacheInfo.total_dates_in_range}개<br/>
                              {cacheInfo.cache_percentage > 0 && (
                                <>⏱️ {cacheInfo.estimated_time_savings}</>
                              )}
                              {cacheInfo.cache_percentage === 100 && (
                                <><br/>✨ <strong>모든 예측이 캐시되어 있어 즉시 완료됩니다!</strong></>
                              )}
                              {cacheInfo.cache_percentage === 0 && (
                                <><br/>🔄 <strong>새로운 예측을 수행합니다</strong></>
                              )}
                            </div>
                          </div>
                        )}
                      </div>
                    );
                  })()}
                  
                  <div style={styles.buttonRow}>
                    <button
                      style={styles.accumulatedButton}
                      onClick={handleStartAccumulatedPrediction}
                      disabled={isPredicting || !selectedStartDate || !endStartDate}
                    >
                      <Activity size={18} />
                      {isPredicting ? '누적 예측 중...' : '누적 예측 시작'}
                    </button>
                  </div>
                </div>
              )}
              
              {/* 진행 상태 표시 */}
              {isPredicting && (
                <div style={styles.progressContainer}>
                  <p style={styles.progressText}>예측 진행 상태: {progress}%</p>
                  <ProgressBar progress={progress} />
                </div>
              )}
              
              {/* 오류 메시지 */}
              {error && (
                <div style={styles.errorMessage}>
                  <AlertTriangle size={16} style={{ marginRight: '0.25rem' }} />
                  {error}
                </div>
              )}
            </div>

            {/* 단일 예측 결과 대시보드 */}
            {activeTab === 'single' && predictionData.length > 0 && (
              <>
                {/* 단일 & 누적 예측 연동 정보 */}
                {currentDate && accumulatedResults && (
                  <div style={styles.card}>
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      backgroundColor: '#f0f9ff',
                      padding: '1rem',
                      borderRadius: '0.5rem',
                      border: '1px solid #bae6fd'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <div style={{ 
                          backgroundColor: '#3b82f6', 
                          borderRadius: '50%', 
                          padding: '0.5rem',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center'
                        }}>
                          <TrendingUp size={16} style={{ color: 'white' }} />
                        </div>
                        <div>
                          <h4 style={{ 
                            margin: 0, 
                            fontSize: '0.875rem', 
                            fontWeight: '600',
                            color: '#1e40af'
                          }}>
                            📊 스마트 캐시 연동 활성화
                          </h4>
                          <p style={{ 
                            margin: 0, 
                            fontSize: '0.75rem', 
                            color: '#6b7280',
                            marginTop: '0.25rem'
                          }}>
                            이 예측 결과({currentDate})는 누적 예측에서도 확인할 수 있습니다
                          </p>
                        </div>
                      </div>
                      <button
                        onClick={() => setActiveTab('accumulated')}
                        style={{
                          backgroundColor: '#8b5cf6',
                          color: 'white',
                          border: 'none',
                          borderRadius: '0.375rem',
                          padding: '0.5rem 1rem',
                          fontSize: '0.75rem',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.25rem'
                        }}
                      >
                        <Activity size={14} />
                        누적 예측으로 이동
                      </button>
                    </div>
                  </div>
                )}
                
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: windowWidth >= 768 ? 'repeat(2, 1fr)' : '1fr',
                  gap: '1.5rem'
                }}>
                <div style={styles.card}>
                  <h2 style={styles.cardTitle}>
                    <TrendingUp size={18} style={styles.iconStyle} />
                    향후 23일 가격 예측
                  </h2>
                  <PredictionChart data={predictionData} />
                </div>            

                {/* 이동평균 차트 */}
                <div style={styles.card}>
                  <h2 style={styles.cardTitle}>
                    <Clock size={18} style={styles.iconStyle} />
                    이동평균 분석 (5일, 10일, 23일)
                  </h2>
                  <MovingAverageChart data={maResults} />
                </div>

                {/* 구간 점수표 */}
                <div style={styles.card}>
                  <h2 style={styles.cardTitle}>
                    <Award size={18} style={styles.iconStyle} />
                    구매 의사결정 구간 점수표
                  </h2>
                  <IntervalScoresTable 
                    data={intervalScores}
                  />
                </div>

                {/* 어텐션 맵 시각화 */}
                <div style={styles.card}>
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '1rem'
                  }}>
                    <h2 style={styles.cardTitle}>
                      <Grid size={18} style={styles.iconStyle} />
                      특성 중요도 시각화 (Attention Map)
                    </h2>
                    <button
                      onClick={handleRefreshAttentionMap}
                      style={{
                        backgroundColor: '#3b82f6',
                        color: 'white',
                        border: 'none',
                        borderRadius: '0.375rem',
                        padding: '0.5rem 1rem',
                        fontSize: '0.875rem',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem'
                      }}
                    >
                      🔄 새로고침
                    </button>
                  </div>
                  <AttentionMap imageData={attentionImage} />
                  <div style={styles.helpText}>
                    <p>* 상위 특성이 MOPJ 예측에 가장 큰 영향을 미치는 요소입니다.</p>
                    {!attentionImage && (
                      <p style={{ color: '#ef4444', fontSize: '0.875rem', marginTop: '0.5rem' }}>
                        * Attention Map이 로드되지 않았습니다. 위의 '새로고침' 버튼을 클릭해주세요.
                      </p>
                    )}
                  </div>
                </div>
              </div>
              </>
            )}
            
            {/* 누적 예측 결과 대시보드 */}
            {activeTab === 'accumulated' && (
              <>
                {/* 로딩 상태 표시 */}
                {isLoading && (
                  <div style={styles.card}>
                    <div style={{
                      textAlign: 'center',
                      padding: '2rem',
                      color: '#6b7280'
                    }}>
                      <Clock size={24} style={{ animation: 'spin 1s linear infinite' }} />
                      <p style={{ marginTop: '0.5rem' }}>누적 예측 결과를 불러오는 중...</p>
                    </div>
                  </div>
                )}

                {/* 데이터 없음 상태 */}
                {!isLoading && !accumulatedResults && (
                  <div style={styles.card}>
                    <div style={{
                      textAlign: 'center',
                      padding: '2rem',
                      color: '#6b7280'
                    }}>
                      <AlertTriangle size={24} />
                      <p style={{ marginTop: '0.5rem' }}>누적 예측 결과가 없습니다. 누적 예측을 먼저 실행해주세요.</p>
                    </div>
                  </div>
                )}

                {/* 실제 결과 표시 */}
                {!isLoading && accumulatedResults && (
                  <>
                    {/* 캐시 클리어 버튼 & 캐시 통계 */}
                    <div style={styles.card}>
                      <div style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: '1rem'
                      }}>
                        <h3>누적 예측 결과</h3>
                        <button
                          onClick={handleClearAccumulatedCache}
                          style={{
                            backgroundColor: '#ef4444',
                            color: 'white',
                            border: 'none',
                            borderRadius: '0.375rem',
                            padding: '0.5rem 1rem',
                            fontSize: '0.875rem',
                            cursor: 'pointer'
                          }}
                        >
                          🧹 캐시 클리어 & 재계산
                        </button>
                      </div>
                      
                      {/* 캐시 통계 정보 표시 */}
                      {accumulatedResults.cache_statistics && (
                        <div style={{
                          backgroundColor: '#f8fafc',
                          border: '1px solid #e2e8f0',
                          borderRadius: '0.375rem',
                          padding: '1rem',
                          marginBottom: '1rem'
                        }}>
                          <h4 style={{ 
                            fontSize: '0.9rem', 
                            fontWeight: '600', 
                            marginBottom: '0.5rem',
                            color: '#374151'
                          }}>
                            🚀 스마트 캐시 활용 현황
                          </h4>
                          <div style={{ fontSize: '0.875rem', color: '#6b7280', lineHeight: '1.5' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span>📊 총 예측 날짜:</span>
                              <span style={{ fontWeight: '600', color: '#059669' }}>
                                {accumulatedResults.cache_statistics.total_dates}개
                              </span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span>⚡ 캐시 활용:</span>
                              <span style={{ fontWeight: '600', color: '#3b82f6' }}>
                                {accumulatedResults.cache_statistics.cached_dates}개 
                                ({accumulatedResults.cache_statistics.cache_hit_rate?.toFixed(1)}%)
                              </span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                              <span>🚀 새로 계산:</span>
                              <span style={{ fontWeight: '600', color: '#f59e0b' }}>
                                {accumulatedResults.cache_statistics.new_predictions}개
                              </span>
                            </div>
                            {accumulatedResults.cache_statistics.cache_hit_rate === 100 && (
                              <div style={{ 
                                marginTop: '0.5rem', 
                                padding: '0.5rem', 
                                backgroundColor: '#d1fae5', 
                                borderRadius: '0.25rem',
                                color: '#065f46',
                                fontWeight: '600'
                              }}>
                                ✨ 모든 예측이 캐시에서 로드되어 즉시 완료되었습니다!
                              </div>
                            )}
                            {accumulatedResults.cache_statistics.cache_hit_rate > 0 && accumulatedResults.cache_statistics.cache_hit_rate < 100 && (
                              <div style={{ 
                                marginTop: '0.5rem', 
                                padding: '0.5rem', 
                                backgroundColor: '#fef3c7', 
                                borderRadius: '0.25rem',
                                color: '#92400e',
                                fontWeight: '600'
                              }}>
                                💡 부분 캐시 활용으로 처리 시간이 단축되었습니다.
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                      
                      <p style={{ fontSize: '0.875rem', color: '#6b7280' }}>
                        구매 신뢰도 계산이 잘못되었다면 위 버튼을 클릭하여 캐시를 클리어하고 다시 누적 예측을 실행해주세요.
                      </p>
                    </div>

                    {/* 신뢰도 종합 분석 카드 */}
                    <ReliabilityAnalysisCard 
                      consistencyScores={consistencyScores ? Object.values(consistencyScores)[0] : null}
                      purchaseReliability={accumulatedResults.accumulated_purchase_reliability || 0}
                      actualBusinessDays={accumulatedResults.predictions ? accumulatedResults.predictions.length : 0}
                    />
                    
                    {/* 누적 예측 요약 */}
                    <AccumulatedSummary 
                      data={accumulatedResults} 
                      onDownloadReport={handleDownloadReport}
                    />
                    
                    {/* 신뢰 날짜 구매 의사결정 구간 카드 */}
                    <div style={styles.card}>
                      <h2 style={styles.cardTitle}>
                        <Award size={18} style={styles.iconStyle} />
                        신뢰 날짜 구매 의사결정 구간
                      </h2>
                      <AccumulatedIntervalScoresTable data={accumulatedResults} />
                    </div>
                    
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: windowWidth >= 768 ? 'repeat(2, 1fr)' : '1fr',
                      gap: '1.5rem'
                    }}>
                      {/* 누적 예측 지표 차트 */}
                      <div style={styles.card}>
                        <h2 style={styles.cardTitle}>
                          <Activity size={18} style={styles.iconStyle} />
                          날짜별 예측 추이
                        </h2>
                        <AccumulatedMetricsChart 
                          data={accumulatedResults}
                        />
                      </div>
                      
                      {/* 누적 예측 결과 테이블 */}
                      <div style={styles.card}>
                        <h2 style={styles.cardTitle}>
                          <BarChart size={18} style={styles.iconStyle} />
                          날짜별 예측 비교
                        </h2>
                        <AccumulatedResultsTable 
                          data={accumulatedResults} 
                          currentDate={selectedAccumulatedDate}
                          onSelectDate={handleAccumulatedDateSelect}
                          onViewInSingle={handleViewInSinglePrediction}
                        />
                      </div>
                      
                      {/* 선택된 날짜의 예측 차트 */}
                      <div style={styles.card}>
                        <div style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'flex-start',
                          marginBottom: '1rem'
                        }}>
                          <h2 style={styles.cardTitle}>
                            <TrendingUp size={18} style={styles.iconStyle} />
                            선택 날짜 ({selectedAccumulatedDate || '없음'}) 예측 결과
                            <span style={{ fontSize: '0.75rem', color: '#6b7280', marginLeft: '0.5rem' }}>
                              (데이터: {selectedDatePredictions?.length || 0}개, 구간: {selectedDateIntervalScores?.length || 0}개)
                              {selectedDatePredictions?.length > 0 && (
                                <span style={{ color: '#3b82f6' }}>
                                  - 첫 예측: {selectedDatePredictions[0]?.Prediction?.toFixed(2) || 'N/A'}
                                </span>
                              )}
                            </span>
                          </h2>
                          
                          {/* 단일 예측으로 보기 버튼 */}
                          {selectedAccumulatedDate && selectedDatePredictions?.length > 0 && (
                            <button
                              onClick={() => handleViewInSinglePrediction(selectedAccumulatedDate)}
                              disabled={isLoading}
                              style={{
                                backgroundColor: '#3b82f6',
                                color: 'white',
                                border: 'none',
                                borderRadius: '0.375rem',
                                padding: '0.5rem 1rem',
                                fontSize: '0.75rem',
                                cursor: isLoading ? 'not-allowed' : 'pointer',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '0.25rem',
                                whiteSpace: 'nowrap',
                                opacity: isLoading ? 0.6 : 1
                              }}
                            >
                              <TrendingUp size={14} />
                              단일 예측으로 보기
                            </button>
                          )}
                        </div>
                        {selectedAccumulatedDate && (!selectedDatePredictions || selectedDatePredictions.length === 0) && (
                          <div style={{
                            padding: '1rem',
                            backgroundColor: '#fef3c7',
                            borderRadius: '0.375rem',
                            marginBottom: '1rem'
                          }}>
                            <p style={{ fontSize: '0.875rem', margin: 0, color: '#92400e' }}>
                              📋 {selectedAccumulatedDate} 날짜를 선택했지만 예측 데이터가 로드되지 않았습니다. 
                              다시 해당 날짜를 클릭해보세요.
                            </p>
                          </div>
                        )}
                        <PredictionChart data={selectedDatePredictions || []} />
                      </div>
                      
                      {/* 선택된 날짜의 구간 점수표 */}
                      <div style={styles.card}>
                        <h2 style={styles.cardTitle}>
                          <Award size={18} style={styles.iconStyle} />
                          선택 날짜 구매 의사결정 구간
                          <span style={{ fontSize: '0.75rem', color: '#6b7280', marginLeft: '0.5rem' }}>
                            ({selectedAccumulatedDate || '없음'} 기준)
                            {selectedDateIntervalScores?.length > 0 && (
                              <span style={{ color: '#10b981' }}>
                                - 첫 구간: {selectedDateIntervalScores[0]?.avg_price?.toFixed(2) || 'N/A'}
                              </span>
                            )}
                          </span>
                        </h2>
                        {selectedAccumulatedDate && (!selectedDateIntervalScores || selectedDateIntervalScores.length === 0) && (
                          <div style={{
                            padding: '1rem',
                            backgroundColor: '#fef3c7',
                            borderRadius: '0.375rem',
                            marginBottom: '1rem'
                          }}>
                            <p style={{ fontSize: '0.875rem', margin: 0, color: '#92400e' }}>
                              📋 {selectedAccumulatedDate} 날짜의 구간 점수 데이터가 로드되지 않았습니다. 
                              다시 해당 날짜를 클릭해보세요.
                            </p>
                          </div>
                        )}
                        <IntervalScoresTable 
                          data={selectedDateIntervalScores || []} 
                          purchaseReliability={accumulatedResults?.accumulated_purchase_reliability || 0}
                        />
                      </div>
                    </div>
                  </>
                )}
              </>
            )}
          </>
        )}
      </main>

      <footer style={styles.footer}>
        © 2025 MOPJ 예측 시스템 | 예측 시작일: {currentDate || '데이터 없음'}
      </footer>
    </div>
      );
  };
  
  export default App;
