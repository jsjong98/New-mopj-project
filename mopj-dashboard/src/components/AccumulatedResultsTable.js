import React from 'react';
import { Calendar, TrendingUp, TrendingDown, Minus, DollarSign } from 'lucide-react';

const styles = {
  container: {
    maxHeight: '16rem',
    overflowY: 'auto'
  },
  noData: {
    height: '16rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f9fafb',
    borderRadius: '0.375rem'
  },
  noDataText: {
    color: '#6b7280'
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '0.875rem'
  },
  tableHeader: {
    position: 'sticky',
    top: 0,
    backgroundColor: '#f9fafb',
    zIndex: 10
  },
  th: {
    padding: '0.75rem 1rem',
    textAlign: 'left',
    fontWeight: '500',
    color: '#6b7280',
    borderBottom: '1px solid #e5e7eb'
  },
  td: {
    padding: '0.5rem 1rem',
    borderBottom: '1px solid #e5e7eb',
    color: '#4b5563'
  },
  date: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem'
  },
  dateIcon: {
    color: '#2563eb'
  },
  good: {
    color: '#10b981',
    display: 'flex',
    alignItems: 'center',
    gap: '0.25rem'
  },
  warning: {
    color: '#f59e0b',
    display: 'flex',
    alignItems: 'center',
    gap: '0.25rem'
  },
  bad: {
    color: '#ef4444',
    display: 'flex',
    alignItems: 'center',
    gap: '0.25rem'
  },
  neutral: {
    color: '#6b7280',
    display: 'flex',
    alignItems: 'center',
    gap: '0.25rem'
  },
  currentRow: {
    backgroundColor: '#f0fdf4',
  },
  buttonContainer: {
    display: 'flex',
    gap: '0.5rem'
  },
  viewButton: {
    backgroundColor: '#3b82f6',
    color: 'white',
    border: 'none',
    padding: '0.25rem 0.5rem',
    borderRadius: '0.25rem',
    fontSize: '0.75rem',
    cursor: 'pointer'
  }
};

// 가격 경향 표시
const getPriceTrend = (predictions) => {
  if (!predictions || predictions.length < 2) {
    return <div style={styles.neutral}><Minus size={14} /> 중립</div>;
  }
  
  const firstPrice = predictions[0].Prediction;
  const lastPrice = predictions[predictions.length - 1].Prediction;
  const change = ((lastPrice - firstPrice) / firstPrice) * 100;
  
  if (change > 3) {
    return <div style={styles.bad}><TrendingUp size={14} /> 상승 ({change.toFixed(1)}%)</div>;
  } else if (change < -3) {
    return <div style={styles.good}><TrendingDown size={14} /> 하락 ({change.toFixed(1)}%)</div>;
  } else {
    return <div style={styles.neutral}><Minus size={14} /> 중립 ({change.toFixed(1)}%)</div>;
  }
};

// 가격 범위 표시
const getPriceRange = (predictions) => {
  if (!predictions || predictions.length === 0) {
    return '-';
  }
  
  const prices = predictions.map(p => p.Prediction);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  
  return `${minPrice.toFixed(2)} ~ ${maxPrice.toFixed(2)}`;
};

// 최적 구매 기간 표시
const getBestInterval = (intervalScores) => {
  if (!intervalScores || !Array.isArray(intervalScores) || intervalScores.length === 0) {
    return null;
  }
  
  // 유효한 항목만 필터링
  const validIntervals = intervalScores.filter(
    interval => interval && typeof interval === 'object' && interval.days != null
  );
  
  if (validIntervals.length === 0) {
    return null;
  }
  
  // 최고 점수 구간 찾기
  return validIntervals.reduce((best, current) => 
    (current.score > best.score) ? current : best
  , validIntervals[0]);
};

const AccumulatedResultsTable = ({ data, currentDate, onSelectDate }) => {
  if (!data || !data.predictions || data.predictions.length === 0) {
    return (
      <div style={styles.noData}>
        <p style={styles.noDataText}>누적 예측 결과 데이터가 없습니다</p>
      </div>
    );
  }

  return (
    <div style={styles.container}>
      <table style={styles.table}>
        <thead style={styles.tableHeader}>
          <tr>
            <th style={styles.th}>기준 날짜</th>
            <th style={styles.th}>가격 범위</th>
            <th style={styles.th}>가격 경향</th>
            <th style={styles.th}>최적 구매 기간</th>
            <th style={styles.th}>작업</th>
          </tr>
        </thead>
        <tbody>
          {data.predictions.map((item) => (
            <tr 
              key={item.date} 
              style={item.date === currentDate ? styles.currentRow : null}
            >
              <td style={styles.td}>
                <div style={styles.date}>
                  <Calendar size={14} style={styles.dateIcon} />
                  {item.date}
                </div>
              </td>
              <td style={styles.td}>{getPriceRange(item.predictions)}</td>
              <td style={styles.td}>{getPriceTrend(item.predictions)}</td>
              <td style={styles.td}>{getBestInterval(item.interval_scores)}</td>
              <td style={styles.td}>
                <div style={styles.buttonContainer}>
                  <button 
                    style={styles.viewButton}
                    onClick={() => onSelectDate(item.date)}
                  >
                    상세 보기
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default AccumulatedResultsTable;