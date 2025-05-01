import React, { useState, useEffect } from 'react';
import { TrendingUp, Calendar, Database, Clock, Grid, Award, RefreshCw, AlertTriangle, BarChart, Activity } from 'lucide-react';
import FileUploader from './components/FileUploader';
import PredictionChart from './components/PredictionChart';
import MovingAverageChart from './components/MovingAverageChart';
import IntervalScoresTable from './components/IntervalScoresTable';
import AttentionMap from './components/AttentionMap';
import ProgressBar from './components/ProgressBar';
import AccumulatedMetricsChart from './components/AccumulatedMetricsChart';
import AccumulatedResultsTable from './components/AccumulatedResultsTable';
import AccumulatedSummary from './components/AccumulatedSummary';
import AccumulatedIntervalScoresTable from './components/AccumulatedIntervalScoresTable';
import HolidayManager from './components/HolidayManager'; // 휴일 관리 컴포넌트 추가
import { 
  startPrediction, 
  getPredictionStatus, 
  getPredictionResults,
  startAccumulatedPrediction,
  getAccumulatedResults,
  getAccumulatedResultByDate,
  getAccumulatedVisualization,
  getAccumulatedReportURL
} from './services/api';

// 앱 전체에서 사용할 스타일 정의
const styles = {
  appContainer: {
    display: 'flex',
    flexDirection: 'column',
    minHeight: '100vh',
    backgroundColor: '#f9fafb'
  },
  header: {
    backgroundColor: '#2563eb',
    color: 'white',
    padding: '1rem',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
  },
  headerContent: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between'
  },
  headerTitle: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem'
  },
  headerInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: '1rem'
  },
  titleText: {
    fontSize: '1.25rem',
    fontWeight: 'bold'
  },
  mainContent: {
    flex: 1,
    padding: '1.5rem'
  },
  card: {
    backgroundColor: 'white',
    borderRadius: '0.5rem',
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
    marginTop: '1rem',
    display: 'flex',
    flexWrap: 'wrap',
    alignItems: 'center',
    gap: '1rem'
  },
  selectContainer: {
    flex: 1,
    minWidth: '250px'
  },
  selectLabel: {
    display: 'block',
    fontSize: '0.875rem',
    fontWeight: '500',
    color: '#374151',
    marginBottom: '0.25rem'
  },
  dateSelect: {
    width: '100%',
    padding: '0.5rem 0.75rem',
    border: '1px solid #d1d5db',
    borderRadius: '0.375rem'
  },
  predictionButton: {
    backgroundColor: '#10b981',
    color: 'white',
    padding: '0.5rem 1rem',
    borderRadius: '0.375rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    cursor: 'pointer',
    border: 'none'
  },
  accumulatedButton: {
    backgroundColor: '#8b5cf6',
    color: 'white',
    padding: '0.5rem 1rem',
    borderRadius: '0.375rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    cursor: 'pointer',
    border: 'none'
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
  }
};

const App = () => {
  // 기본 상태 관리
  const [fileInfo, setFileInfo] = useState(null);
  const [selectedDate, setSelectedDate] = useState(null);
  const [endDate, setEndDate] = useState(null); // 누적 예측 종료 날짜
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
  
  // 탭 관리
  const [activeTab, setActiveTab] = useState('single'); // 'single' 또는 'accumulated'
  
  // 시스템 탭 관리 ('prediction' 또는 'settings')
  const [systemTab, setSystemTab] = useState('prediction');
  
  // 누적 예측 관련 상태
  const [accumulatedResults, setAccumulatedResults] = useState(null);
  const [accumulatedVisualization, setAccumulatedVisualization] = useState(null);
  const [selectedAccumulatedDate, setSelectedAccumulatedDate] = useState(null);

  // 핸들러 함수
  const handleUploadSuccess = (data) => {
    setFileInfo(data);
    setSelectedDate(data.latestDate);
    setEndDate(data.latestDate); // 기본적으로 마지막 날짜로 설정
    setIsCSVUploaded(true);
    setError(null);
  };

  // 단일 예측 시작
  const handleStartPrediction = async () => {
    if (!fileInfo || !fileInfo.filepath) {
      setError('파일을 먼저 업로드해주세요.');
      return;
    }

    setError(null);
    setIsPredicting(true);
    setProgress(0);

    try {
      await startPrediction(fileInfo.filepath, selectedDate);
      checkPredictionStatus();
    } catch (err) {
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

      if (!selectedDate) {
        setError('시작 날짜를 선택해주세요.');
        return;
      }

      setError(null);
      setIsPredicting(true);
      setProgress(0);
      
      // 디버그 로그 추가
      console.log("Starting accumulated prediction:", {
        filepath: fileInfo.filepath,
        startDate: selectedDate,
        endDate: endDate
      });

      try {
        await startAccumulatedPrediction(fileInfo.filepath, selectedDate, endDate);
        checkPredictionStatus('accumulated');
      } catch (err) {
        setError(err.error || '누적 예측 시작 중 오류가 발생했습니다.');
        setIsPredicting(false);
      }
  };

  // 예측 상태 확인
  const checkPredictionStatus = (mode = 'single') => {
    const statusInterval = setInterval(async () => {
      try {
        const status = await getPredictionStatus();
        
        setProgress(status.progress);
        
        if (!status.is_predicting) {
          clearInterval(statusInterval);
          setIsPredicting(false);
          
          if (status.error) {
            setError(`예측 오류: ${status.error}`);
          } else {
            if (mode === 'accumulated') {
              fetchAccumulatedResults();
            } else {
              fetchResults();
            }
          }
        }
      } catch (err) {
        clearInterval(statusInterval);
        setIsPredicting(false);
        setError('예측 상태 확인 중 오류가 발생했습니다.');
      }
    }, 1000);
  };

  // 단일 예측 결과 가져오기
  const fetchResults = async () => {
    setIsLoading(true);
    
    try {
      const results = await getPredictionResults();
      
      setPredictionData(results.predictions || []);
      setIntervalScores(results.interval_scores || []);
      setMaResults(results.ma_results || null);
      setAttentionImage(results.attention_data ? results.attention_data.image : null);
      setCurrentDate(results.current_date || null);
      
      // 단일 예측 탭으로 전환
      setActiveTab('single');
    } catch (err) {
      setError('결과를 가져오는 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  // 누적 예측 결과 가져오기
  const fetchAccumulatedResults = async () => {
    setIsLoading(true);
    
    try {
      const results = await getAccumulatedResults();
      
      if (results.success) {
        setAccumulatedResults(results);
        
        // 시각화 데이터 가져오기
        try {
          const visualizationData = await getAccumulatedVisualization();
          if (visualizationData.success) {
            setAccumulatedVisualization(visualizationData);
          }
        } catch (vizErr) {
          console.error('누적 시각화 로드 오류:', vizErr);
        }
        
        // 가장 최근 날짜의 예측 결과 선택
        if (results.predictions && results.predictions.length > 0) {
          const latestPrediction = results.predictions[results.predictions.length - 1];
          setSelectedAccumulatedDate(latestPrediction.date);
          loadSelectedDatePrediction(latestPrediction.date);
        }
        
        // 누적 예측 탭으로 전환
        setActiveTab('accumulated');
      } else {
        setError(results.error || '누적 예측 결과가 없습니다.');
      }
    } catch (err) {
      setError('누적 결과를 가져오는 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  // 특정 날짜의 예측 결과 로드
  const loadSelectedDatePrediction = async (date) => {
    if (!date) return;
    
    setIsLoading(true);
    try {
      const result = await getAccumulatedResultByDate(date);
      
      if (result.success) {
        setPredictionData(result.predictions || []);
        setIntervalScores(result.interval_scores || []);
        setCurrentDate(result.date || null);
        
        // 이 날짜에 대한 MA 결과와 어텐션 맵도 있다면 로드
        // (여기서는 데이터가 없다고 가정하고 비워둠)
        setMaResults(null);
        setAttentionImage(null);
      } else {
        setError(`${date} 날짜의 결과를 로드할 수 없습니다.`);
      }
    } catch (err) {
      setError(`날짜 데이터 로드 오류: ${err.message || '알 수 없는 오류'}`);
    } finally {
      setIsLoading(false);
    }
  };

  // 보고서 다운로드
  const handleDownloadReport = () => {
    const reportUrl = getAccumulatedReportURL();
    window.open(reportUrl, '_blank');
  };

  // 새로고침 처리
  const handleRefresh = () => {
    if (fileInfo && fileInfo.filepath) {
      if (activeTab === 'accumulated') {
        handleStartAccumulatedPrediction();
      } else {
        handleStartPrediction();
      }
    }
  };

  // 누적 예측 날짜 선택 시
  const handleAccumulatedDateSelect = (date) => {
    setSelectedAccumulatedDate(date);
    loadSelectedDatePrediction(date);
  };

  return (
    <div style={styles.appContainer}>
      {/* 헤더 */}
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <div style={styles.headerTitle}>
            <TrendingUp size={24} />
            <h1 style={styles.titleText}>MOPJ 예측 및 구매 전략 대시보드</h1>
          </div>
          <div style={styles.headerInfo}>
            {currentDate && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Calendar size={18} />
                <span>기준일: {currentDate}</span>
              </div>
            )}
          </div>
        </div>
      </header>

      <main style={styles.mainContent}>
        {/* 시스템 탭 선택 */}
        <div style={styles.tabContainer}>
          <div 
            style={styles.tab(systemTab === 'prediction')}
            onClick={() => setSystemTab('prediction')}
          >
            <TrendingUp size={16} />
            예측 시스템
          </div>
          <div 
            style={styles.tab(systemTab === 'settings')}
            onClick={() => setSystemTab('settings')}
          >
            <Calendar size={16} />
            휴일 설정
          </div>
        </div>

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
              
              {/* 탭 선택 */}
              {isCSVUploaded && (
                <div style={styles.tabContainer}>
                  <div 
                    style={styles.tab(activeTab === 'single')}
                    onClick={() => setActiveTab('single')}
                  >
                    <TrendingUp size={16} />
                    단일 날짜 예측
                  </div>
                  <div 
                    style={styles.tab(activeTab === 'accumulated')}
                    onClick={() => setActiveTab('accumulated')}
                  >
                    <Activity size={16} />
                    누적 예측 분석
                  </div>
                </div>
              )}
              
              {/* 파일 업로드 컴포넌트 */}
              {!isCSVUploaded && (
                <FileUploader 
                  onUploadSuccess={handleUploadSuccess}
                  isLoading={isLoading}
                  setIsLoading={setIsLoading}
                />
              )}
              
              {/* 단일 예측 날짜 선택 */}
              {isCSVUploaded && activeTab === 'single' && (
                <div style={styles.dateSelectContainer}>
                  <div style={styles.selectContainer}>
                    <label htmlFor="date-select" style={styles.selectLabel}>
                      기준 날짜 선택
                    </label>
                    <select
                      id="date-select"
                      style={styles.dateSelect}
                      value={selectedDate || ''}
                      onChange={(e) => setSelectedDate(e.target.value)}
                      disabled={isPredicting}
                    >
                      {fileInfo.dates.map((date) => (
                        <option key={date} value={date}>{date}</option>
                      ))}
                    </select>
                  </div>
                  
                  <button
                    style={styles.predictionButton}
                    onClick={handleStartPrediction}
                    disabled={isPredicting || !selectedDate}
                  >
                    <TrendingUp size={18} />
                    {isPredicting ? '예측 중...' : '예측 시작'}
                  </button>
                </div>
              )}
              
              {/* 누적 예측 날짜 선택 */}
              {isCSVUploaded && activeTab === 'accumulated' && (
                <div style={styles.dateSelectContainer}>
                  <div style={styles.selectContainer}>
                    <label htmlFor="start-date-select" style={styles.selectLabel}>
                      시작 날짜
                    </label>
                    <select
                      id="start-date-select"
                      style={styles.dateSelect}
                      value={selectedDate || ''}
                      onChange={(e) => setSelectedDate(e.target.value)}
                      disabled={isPredicting}
                    >
                      {fileInfo.dates.map((date) => (
                        <option key={date} value={date}>{date}</option>
                      ))}
                    </select>
                  </div>
                  
                  <div style={styles.selectContainer}>
                    <label htmlFor="end-date-select" style={styles.selectLabel}>
                      종료 날짜
                    </label>
                    <select
                      id="end-date-select"
                      style={styles.dateSelect}
                      value={endDate || ''}
                      onChange={(e) => setEndDate(e.target.value)}
                      disabled={isPredicting}
                    >
                      {fileInfo.dates
                        .filter(date => date >= selectedDate)
                        .map((date) => (
                        <option key={date} value={date}>{date}</option>
                      ))}
                    </select>
                  </div>
                  
                  <button
                    style={styles.accumulatedButton}
                    onClick={handleStartAccumulatedPrediction}
                    disabled={isPredicting || !selectedDate || !endDate}
                  >
                    <Activity size={18} />
                    {isPredicting ? '누적 예측 중...' : '누적 예측 시작'}
                  </button>
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
              <div style={{
                display: 'grid',
                gridTemplateColumns: window.innerWidth >= 768 ? 'repeat(2, 1fr)' : '1fr',
                gap: '1.5rem'
              }}>
                {/* 가격 예측 차트 */}
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
                  <IntervalScoresTable data={intervalScores} />
                </div>

                {/* 어텐션 맵 시각화 */}
                <div style={styles.card}>
                  <h2 style={styles.cardTitle}>
                    <Grid size={18} style={styles.iconStyle} />
                    특성 중요도 시각화 (Attention Map)
                  </h2>
                  <AttentionMap imageData={attentionImage} />
                  <div style={styles.helpText}>
                    <p>* 상위 특성이 MOPJ 예측에 가장 큰 영향을 미치는 요소입니다.</p>
                  </div>
                </div>
              </div>
            )}
            
            {/* 누적 예측 결과 대시보드 */}
            {activeTab === 'accumulated' && accumulatedResults && (
              <>
                {/* 누적 예측 요약 */}
                <AccumulatedSummary 
                  data={accumulatedResults} 
                  onDownloadReport={handleDownloadReport}
                />
                
                {/* 신뢰 날짜 구매 의사결정 구간 카드 추가 */}
                <div style={styles.card}>
                  <h2 style={styles.cardTitle}>
                    <Award size={18} style={styles.iconStyle} />
                    신뢰 날짜 구매 의사결정 구간
                  </h2>
                  <AccumulatedIntervalScoresTable data={accumulatedResults} />
                </div>
                
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: window.innerWidth >= 768 ? 'repeat(2, 1fr)' : '1fr',
                  gap: '1.5rem'
                }}>
                  {/* 누적 예측 지표 차트 */}
                  <div style={styles.card}>
                    <h2 style={styles.cardTitle}>
                      <Activity size={18} style={styles.iconStyle} />
                      날짜별 예측 추이
                    </h2>
                    <AccumulatedMetricsChart 
                      data={{
                        metrics: accumulatedResults.predictions
                      }}
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
                    />
                  </div>
                  
                  {/* 선택된 날짜의 예측 차트 */}
                  <div style={styles.card}>
                    <h2 style={styles.cardTitle}>
                      <TrendingUp size={18} style={styles.iconStyle} />
                      선택 날짜 ({selectedAccumulatedDate || '없음'}) 예측 결과
                    </h2>
                    <PredictionChart data={predictionData} />
                  </div>
                  
                  {/* 선택된 날짜의 구간 점수표 */}
                  <div style={styles.card}>
                    <h2 style={styles.cardTitle}>
                      <Award size={18} style={styles.iconStyle} />
                      선택 날짜 구매 의사결정 구간
                    </h2>
                    <IntervalScoresTable data={intervalScores} />
                  </div>
                </div>
              </>
            )}
          </>
        )}
      </main>

      <footer style={styles.footer}>
        © 2025 MOPJ 예측 시스템 | 데이터 기준: {currentDate || '데이터 없음'}
      </footer>
    </div>
  );
};

export default App;
