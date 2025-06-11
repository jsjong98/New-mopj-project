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

const PredictionChart = ({ data, title }) => {
  if (!data || data.length === 0) {
    return (
      <div style={styles.noDataContainer}>
        <p style={styles.noDataText}>데이터가 없습니다</p>
      </div>
    );
  }

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

  return (
    <div style={styles.chartContainer}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="Date" 
            tick={{ fontSize: 12 }}
            tickFormatter={formatDate}
          />
          <YAxis domain={['auto', 'auto']} />
          <Tooltip
            formatter={(value, name) => {
              if (value === null || value === undefined) {
                return ['데이터 없음', name === "Prediction" ? "예측 가격" : "실제 가격"];
              }
              return [
                `${parseFloat(value).toFixed(2)}`,
                name === "Prediction" ? "예측 가격" : "실제 가격"
              ];
            }}
            labelFormatter={(label) => `날짜: ${formatDate(label)}`}
          />
          <Legend />
          {/* 실제 가격 라인 (데이터가 있을 때만 표시) */}
          {data.some(item => item.Actual !== null && item.Actual !== undefined) && (
            <Line 
              type="monotone" 
              dataKey="Actual" 
              stroke="#3b82f6" 
              strokeWidth={2} 
              name="실제 가격" 
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
            />
          )}
          
          {/* 예측 가격 라인 (항상 표시) */}
          <Line 
            type="monotone" 
            dataKey="Prediction" 
            stroke="#ef4444" 
            strokeWidth={2} 
            name="예측 가격" 
            dot={{ r: 4 }}
            strokeDasharray="5 5"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default PredictionChart;
