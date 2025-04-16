// api.js
import axios from 'axios';

// 백엔드 URL 직접 지정
const API_BASE_URL = 'http://localhost:5000';

// API 클라이언트 생성
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json'
  },
  timeout: 60000 // 60초 타임아웃
});

// CSV 파일 업로드 - fetch API 사용
export const uploadCSV = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    // 직접 백엔드 URL로 요청
    const response = await fetch(`${API_BASE_URL}/api/upload`, {
      method: 'POST',
      body: formData,
      // CORS 설정
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('Upload failed:', errorText);
      throw new Error('파일 업로드 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Upload error:', error);
    return { error: error.message || '파일 업로드 중 오류가 발생했습니다.' };
  }
};

// 사용 가능한 날짜 조회
export const getAvailableDates = async (filepath) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/data/dates?filepath=${encodeURIComponent(filepath)}`, {
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('날짜 정보를 가져오는데 실패했습니다');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Get dates error:', error);
    return { 
      error: error.message || '날짜 정보를 가져오는 중 오류가 발생했습니다.',
      dates: [] 
    };
  }
};

// 예측 시작
export const startPrediction = async (filepath, date = null) => {
  try {
    const payload = { filepath };
    if (date) payload.date = date;
    
    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload),
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('예측 시작 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Start prediction error:', error);
    return { error: error.message || '예측 시작 중 오류가 발생했습니다.' };
  }
};

// 예측 상태 확인
export const getPredictionStatus = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/predict/status`, {
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('예측 상태 확인 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Get prediction status error:', error);
    return { error: error.message || '예측 상태 확인 중 오류가 발생했습니다.' };
  }
};

// 모든 예측 결과 조회
export const getPredictionResults = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/results`, {
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('예측 결과 조회 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Get prediction results error:', error);
    return { error: error.message || '예측 결과를 가져오는 중 오류가 발생했습니다.' };
  }
};

// 예측값만 조회
export const getPredictions = async () => {
  try {
    const response = await apiClient.get('/results/predictions');
    return response.data;
  } catch (error) {
    console.error('Get predictions error:', error);
    throw error.response?.data || { error: error.message || '예측값을 가져오는 중 오류가 발생했습니다.' };
  }
};

// 구간 점수 조회
export const getIntervalScores = async () => {
  try {
    const response = await apiClient.get('/results/interval-scores');
    return response.data;
  } catch (error) {
    console.error('Get interval scores error:', error);
    throw error.response?.data || { error: error.message || '구간 점수를 가져오는 중 오류가 발생했습니다.' };
  }
};

// 이동평균 조회
export const getMovingAverages = async () => {
  try {
    const response = await apiClient.get('/results/moving-averages');
    return response.data;
  } catch (error) {
    console.error('Get moving averages error:', error);
    throw error.response?.data || { error: error.message || '이동평균을 가져오는 중 오류가 발생했습니다.' };
  }
};

// 어텐션 맵 조회
export const getAttentionMap = async () => {
  try {
    const response = await apiClient.get('/results/attention-map');
    return response.data;
  } catch (error) {
    console.error('Get attention map error:', error);
    throw error.response?.data || { error: error.message || '어텐션 맵을 가져오는 중 오류가 발생했습니다.' };
  }
};

// 여기서부터 추가된 누적 예측 관련 함수들입니다

// 누적 예측 시작
export const startAccumulatedPrediction = async (filepath, startDate, endDate) => {
  try {
    const payload = { 
      filepath,
      start_date: startDate,
      end_date: endDate
    };
    
    const response = await fetch(`${API_BASE_URL}/api/predict/accumulated`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload),
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('누적 예측 시작 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Start accumulated prediction error:', error);
    return { error: error.message || '누적 예측 시작 중 오류가 발생했습니다.' };
  }
};

// 특정 날짜의 누적 예측 결과 조회
export const getAccumulatedResultByDate = async (date) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/results/accumulated/${date}`, {
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error(`${date} 날짜의 누적 예측 결과 조회 실패`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Get accumulated result by date error:', error);
    return { error: error.message || '특정 날짜의 누적 예측 결과를 가져오는 중 오류가 발생했습니다.' };
  }
};

// 누적 예측 시각화 데이터 조회
export const getAccumulatedVisualization = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/results/accumulated/visualization`, {
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('누적 예측 시각화 데이터 조회 실패');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Get accumulated visualization error:', error);
    return { error: error.message || '누적 예측 시각화 데이터를 가져오는 중 오류가 발생했습니다.' };
  }
};

// api.js - getAccumulatedResults 함수 디버깅 로그 추가
export const getAccumulatedResults = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/results/accumulated`, {
      mode: 'cors',
      credentials: 'omit'
    });
    
    if (!response.ok) {
      throw new Error('누적 예측 결과 조회 실패');
    }
    
    const data = await response.json();
    console.log('Accumulated results data:', data); // 디버깅 로그 추가
    
    // 확인: accumulated_interval_scores가 있는지
    if (data.success && !data.accumulated_interval_scores) {
      console.warn('Warning: No accumulated_interval_scores in response');
    }
    
    return data;
  } catch (error) {
    console.error('Get accumulated results error:', error);
    return { error: error.message || '누적 예측 결과를 가져오는 중 오류가 발생했습니다.' };
  }
};

// 누적 예측 보고서 URL 가져오기
export const getAccumulatedReportURL = () => {
  return `${API_BASE_URL}/api/results/accumulated/report`;
};
