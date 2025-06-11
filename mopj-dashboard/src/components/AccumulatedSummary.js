import React from 'react';
import { Award, LineChart, AlertTriangle, ArrowDownCircle, TrendingUp, TrendingDown, DollarSign, Minus } from 'lucide-react';

const styles = {
  container: {
    padding: '1rem',
    backgroundColor: '#f0fdf4',
    borderRadius: '0.5rem',
    marginBottom: '1rem'
  },
  noData: {
    padding: '1rem',
    backgroundColor: '#f9fafb',
    borderRadius: '0.5rem',
    color: '#6b7280',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: '1rem'
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '0.75rem'
  },
  title: {
    fontSize: '1rem',
    fontWeight: '600',
    color: '#166534',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem'
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '1rem'
  },
  metricsCard: {
    backgroundColor: 'white',
    borderRadius: '0.375rem',
    padding: '0.75rem',
    boxShadow: '0 1px 2px rgba(0, 0, 0, 0.05)'
  },
  cardHeader: {
    fontSize: '0.875rem',
    color: '#6b7280',
    marginBottom: '0.5rem'
  },
  metricValue: {
    fontSize: '1.5rem',
    fontWeight: '600',
    color: '#166534'
  },
  downloadButton: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    backgroundColor: '#2563eb',
    color: 'white',
    padding: '0.375rem 0.75rem',
    borderRadius: '0.375rem',
    fontSize: '0.875rem',
    border: 'none',
    cursor: 'pointer'
  },
  warning: {
    color: '#d97706',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    fontSize: '0.875rem',
    marginTop: '0.75rem'
  }
};

const AccumulatedSummary = ({ data, onDownloadReport }) => {
  if (!data || !data.predictions || !data.predictions.length) {
    return (
      <div style={styles.noData}>
        <AlertTriangle size={16} style={{ marginRight: '0.5rem' }} />
        누적 예측 데이터가 없습니다
      </div>
    );
  }

  // 모든 예측에서 가격 범위와 추세 계산
  let totalMinPrice = 0;
  let totalMaxPrice = 0;
  let totalPriceRange = 0;
  let totalBestIntervalDays = 0;
  let intervalCount = 0; // 유효한 interval_scores를 가진 예측의 수
  let uptrends = 0;
  let downtrends = 0;
  let total = data.predictions.length;

  data.predictions.forEach(pred => {
    // 안전한 predictions 데이터 접근
    const predictions = Array.isArray(pred.predictions) ? pred.predictions : [];
    const prices = predictions.map(p => p.Prediction || p.prediction).filter(price => typeof price === 'number' && !isNaN(price));
    
    if (prices.length > 0) {
      const minPrice = Math.min(...prices);
      const maxPrice = Math.max(...prices);
      totalMinPrice += minPrice;
      totalMaxPrice += maxPrice;
      totalPriceRange += (maxPrice - minPrice);

      // 추세 계산
      if (prices.length >= 2) {
        const firstPrice = prices[0];
        const lastPrice = prices[prices.length - 1];
        const change = ((lastPrice - firstPrice) / firstPrice) * 100;
        
        if (change > 3) uptrends++;
        else if (change < -3) downtrends++;
      }
    }

    // 최적 구매 기간
    if (pred.interval_scores) {
      if (Array.isArray(pred.interval_scores) && pred.interval_scores.length > 0) {
        // 배열인 경우
        const bestInterval = pred.interval_scores[0];
        if (bestInterval && bestInterval.days) {
          totalBestIntervalDays += bestInterval.days;
          intervalCount++;
        }
      } else if (typeof pred.interval_scores === 'object' && Object.keys(pred.interval_scores).length > 0) {
        // 객체인 경우
        const intervalScoresList = Object.values(pred.interval_scores);
        if (intervalScoresList.length > 0) {
          // 점수가 가장 높은 구간 찾기
          const bestInterval = intervalScoresList.reduce((best, current) => {
            return (!best || (current && current.score > best.score)) ? current : best;
          }, null);
          
          if (bestInterval && bestInterval.days) {
            totalBestIntervalDays += bestInterval.days;
            intervalCount++;
          }
        }
      }
    }
  });

  // 평균 계산
  const avgMinPrice = totalMinPrice / total;
  const avgMaxPrice = totalMaxPrice / total;
  const avgPriceRange = totalPriceRange / total;
  const avgBestIntervalDays = intervalCount > 0 ? totalBestIntervalDays / intervalCount : 0;
  
  // 추세 분포
  const uptrendPercent = (uptrends / total) * 100;
  const downtrendPercent = (downtrends / total) * 100;
  const neutralPercent = 100 - uptrendPercent - downtrendPercent;

  // 추세 아이콘 및 텍스트 결정
  let trendIcon = <Minus size={18} />;
  let trendText = "중립적";
  let trendColor = "#6b7280";
  
  if (uptrendPercent > downtrendPercent && uptrendPercent > neutralPercent) {
    trendIcon = <TrendingUp size={18} />;
    trendText = "상승";
    trendColor = "#ef4444";
  } else if (downtrendPercent > uptrendPercent && downtrendPercent > neutralPercent) {
    trendIcon = <TrendingDown size={18} />;
    trendText = "하락";
    trendColor = "#10b981";
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>
          <Award size={18} />
          누적 예측 결과 요약
        </h3>
        <button 
          style={styles.downloadButton}
          onClick={onDownloadReport}
        >
          <ArrowDownCircle size={16} />
          보고서 다운로드
        </button>
      </div>
      
      <div style={styles.grid}>
        <div style={styles.metricsCard}>
          <div style={styles.cardHeader}>평균 가격 범위</div>
          <div style={styles.metricValue}>{avgMinPrice.toFixed(2)} ~ {avgMaxPrice.toFixed(2)}</div>
        </div>
        
        <div style={styles.metricsCard}>
          <div style={styles.cardHeader}>주요 가격 추세</div>
          <div style={{...styles.metricValue, color: trendColor, display: 'flex', alignItems: 'center', gap: '0.25rem'}}>
            {trendIcon} {trendText} ({Math.max(uptrendPercent, downtrendPercent, neutralPercent).toFixed(0)}%)
          </div>
        </div>
        
        <div style={styles.metricsCard}>
          <div style={styles.cardHeader}>평균 변동폭</div>
          <div style={styles.metricValue}>{avgPriceRange.toFixed(2)}</div>
        </div>
        
        <div style={styles.metricsCard}>
          <div style={styles.cardHeader}>평균 최적 구매 기간</div>
          <div style={styles.metricValue}>
            <DollarSign size={18} style={{display: 'inline', marginRight: '0.25rem'}} />
            {avgBestIntervalDays > 0 ? avgBestIntervalDays.toFixed(1) + '일' : '0.0일'}
          </div>
        </div>
      </div>
      
      <div style={styles.warning}>
        <LineChart size={16} />
        총 {total}개 날짜에 대한 예측 데이터가 누적되었습니다. 이 정보를 바탕으로 최적의 구매 전략을 세우세요.
      </div>
    </div>
  );
};

export default AccumulatedSummary;
