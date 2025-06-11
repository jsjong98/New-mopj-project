import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const styles = {
  chartContainer: {
    height: '16rem'
  },
  noDataContainer: {
    height: '16rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f9fafb',
    borderRadius: '0.375rem'
  },
  noDataText: {
    color: '#6b7280'
  }
};

const AccumulatedMetricsChart = ({ data }) => {
  if (!data || !data.predictions || data.predictions.length === 0) {
    return (
      <div style={styles.noDataContainer}>
        <p style={styles.noDataText}>누적 예측 데이터가 없습니다</p>
      </div>
    );
  }

  // 데이터 형식 변환
  const chartData = data.predictions.map(item => {
    // 안전한 예측 가격 범위와 추세 계산
    const predictions = Array.isArray(item.predictions) ? item.predictions : [];
    const prices = predictions.map(p => p.Prediction || p.prediction).filter(price => typeof price === 'number' && !isNaN(price));
    
    let minPrice = 0, maxPrice = 0, priceRange = 0, changePercent = 0;
    
    if (prices.length > 0) {
      minPrice = Math.min(...prices);
      maxPrice = Math.max(...prices);
      priceRange = maxPrice - minPrice;
      
      if (prices.length >= 2) {
        const firstPrice = prices[0];
        const lastPrice = prices[prices.length - 1];
        changePercent = ((lastPrice - firstPrice) / firstPrice) * 100;
      }
    }
    
    // 최적 구매 기간
    let bestInterval = 0;
    if (item.interval_scores) {
      if (Array.isArray(item.interval_scores) && item.interval_scores.length > 0) {
        // 배열인 경우
        bestInterval = item.interval_scores[0].days || 0;
      } else if (typeof item.interval_scores === 'object' && Object.keys(item.interval_scores).length > 0) {
        // 객체인 경우 - 최고 점수 구간 찾기
        const intervalScoresList = Object.values(item.interval_scores);
        const bestIntervalData = intervalScoresList.reduce((best, current) => {
          return (!best || (current && current.score > best.score)) ? current : best;
        }, null);
        
        if (bestIntervalData && bestIntervalData.days) {
          bestInterval = bestIntervalData.days;
        }
      }
    }
    
    return {
      date: item.date,
      minPrice,
      maxPrice,
      priceRange,
      changePercent,
      bestInterval
    };
  });

  // 날짜 형식화 함수
  const formatDate = (dateString) => {
    if (!dateString) return '';
    
    // 이미 YYYY-MM-DD 형식이면 그대로 반환
    if (/^\d{4}-\d{2}-\d{2}$/.test(dateString)) {
      return dateString;
    }
    
    // GMT 포함된 문자열이면 파싱하여 변환
    if (dateString.includes('GMT')) {
      const date = new Date(dateString);
      return date.toISOString().split('T')[0];
    }
    
    // 기타 경우 처리
    try {
      const date = new Date(dateString);
      return date.toISOString().split('T')[0];
    } catch (e) {
      console.error('날짜 변환 오류:', e);
      return dateString;
    }
  };

  // 툴팁 커스텀 포맷터
  const customTooltipFormatter = (value, name) => {
    let displayName = name;
    let formattedValue = value;
    
    if (name === 'minPrice') displayName = '최저가';
    else if (name === 'maxPrice') displayName = '최고가';
    else if (name === 'priceRange') displayName = '가격 범위';
    else if (name === 'changePercent') {
      displayName = '변동률';
      formattedValue = `${value.toFixed(2)}%`;
      return [formattedValue, displayName];
    }
    else if (name === 'bestInterval') {
      displayName = '최적 구매 기간';
      formattedValue = `${value}일`;
      return [formattedValue, displayName];
    }
    
    return [formattedValue.toFixed(2), displayName];
  };

  return (
    <div style={styles.chartContainer}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="date" 
            tick={{ fontSize: 12 }}
            tickFormatter={formatDate}
          />
          <YAxis 
            yAxisId="price"
            domain={['auto', 'auto']} 
            tick={{ fontSize: 12 }}
          />
          <YAxis 
            yAxisId="percent" 
            orientation="right" 
            domain={[-10, 10]}
            tick={{ fontSize: 12 }}
          />
          <Tooltip 
            formatter={customTooltipFormatter}
            labelFormatter={(label) => `날짜: ${formatDate(label)}`}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="minPrice" 
            stroke="#10b981" 
            strokeWidth={2} 
            name="최저가" 
            dot={{ r: 4 }}
            yAxisId="price"
          />
          <Line 
            type="monotone" 
            dataKey="maxPrice" 
            stroke="#ef4444" 
            strokeWidth={2} 
            name="최고가" 
            dot={{ r: 4 }}
            yAxisId="price"
          />
          <Line 
            type="monotone" 
            dataKey="changePercent" 
            stroke="#8b5cf6" 
            strokeWidth={2} 
            name="변동률 (%)" 
            dot={{ r: 4 }}
            yAxisId="percent"
            strokeDasharray="5 5"
          />
          <Line 
            type="monotone" 
            dataKey="bestInterval" 
            stroke="#3b82f6" 
            strokeWidth={2} 
            name="최적 구매 기간 (일)" 
            dot={{ r: 4 }}
            yAxisId="price"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default AccumulatedMetricsChart;