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

const MovingAverageChart = ({ data }) => {
  // MA 데이터 구조 변환
  const transformData = () => {
    if (!data || !data.ma5 || !data.ma10 || !data.ma23) {
      return [];
    }
    
    // ma5 데이터 기준으로 병합
    return data.ma5.map((item, index) => {
      const ma10Value = index < data.ma10.length ? data.ma10[index].ma : null;
      const ma23Value = index < data.ma23.length ? data.ma23[index].ma : null;
      
      return {
        date: item.date,
        prediction: item.prediction,
        actual: item.actual,
        ma5: item.ma,
        ma10: ma10Value,
        ma23: ma23Value
      };
    });
  };

  const chartData = transformData();

  if (chartData.length === 0) {
    return (
      <div style={styles.noDataContainer}>
        <p style={styles.noDataText}>이동평균 데이터가 없습니다</p>
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
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="date" 
            tick={{ fontSize: 12 }}
            tickFormatter={formatDate}
          />
          <YAxis domain={['auto', 'auto']} />
          <Tooltip
            formatter={(value, name) => {
              if (value === null) return ['데이터 없음', name];
              let displayName;
              switch(name) {
                case "prediction":
                  displayName = "예측 가격";
                  break;
                case "actual":
                  displayName = "실제 가격";
                  break;
                case "ma5":
                  displayName = "5일 이동평균";
                  break;
                case "ma10":
                  displayName = "10일 이동평균";
                  break;
                case "ma23":
                  displayName = "23일 이동평균";
                  break;
                default:
                  displayName = name;  // 알 수 없는 경우 원본 이름 표시
              }
              return [`${parseFloat(value).toFixed(2)}`, displayName];
            }}
            labelFormatter={(label) => `날짜: ${formatDate(label)}`}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="prediction" 
            stroke="#ef4444" 
            strokeWidth={1.5} 
            name="예측 가격" 
            dot={false}
            opacity={0.5}
          />
          <Line 
            type="monotone" 
            dataKey="ma5" 
            stroke="#3b82f6" 
            strokeWidth={2.5} 
            name="5일 이동평균" 
            dot={false}
          />
          <Line 
            type="monotone" 
            dataKey="ma10" 
            stroke="#16a34a" 
            strokeWidth={2.5} 
            name="10일 이동평균" 
            dot={false}
          />
          <Line 
            type="monotone" 
            dataKey="ma23" 
            stroke="#9333ea" 
            strokeWidth={2.5} 
            name="23일 이동평균" 
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default MovingAverageChart;
