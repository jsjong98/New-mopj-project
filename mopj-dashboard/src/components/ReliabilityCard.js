import React from 'react';
import { Shield, TrendingUp, Star, AlertTriangle } from 'lucide-react';

const styles = {
  card: {
    backgroundColor: 'white',
    borderRadius: '0.5rem',
    padding: '1rem',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)'
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    marginBottom: '1rem'
  },
  title: {
    fontSize: '1rem',
    fontWeight: '600',
    color: '#374151',
    marginLeft: '0.5rem'
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '1rem'
  },
  metric: {
    textAlign: 'center',
    padding: '0.75rem',
    backgroundColor: '#f9fafb',
    borderRadius: '0.375rem'
  },
  metricValue: (score) => ({
    fontSize: '1.5rem',
    fontWeight: 'bold',
    color: score >= 75 ? '#10b981' : score >= 50 ? '#f59e0b' : '#ef4444'
  }),
  metricLabel: {
    fontSize: '0.875rem',
    color: '#6b7280',
    marginTop: '0.25rem'
  },
  grade: (grade) => ({
    display: 'inline-flex',
    alignItems: 'center',
    gap: '0.25rem',
    padding: '0.25rem 0.5rem',
    borderRadius: '0.25rem',
    fontSize: '0.75rem',
    fontWeight: '500',
    backgroundColor: getGradeColor(grade).bg,
    color: getGradeColor(grade).text
  }),
  noData: {
    textAlign: 'center',
    color: '#6b7280',
    padding: '2rem'
  }
};

const getGradeColor = (grade) => {
  switch (grade) {
    case 'Very High':
      return { bg: '#d1fae5', text: '#065f46' };
    case 'High':
      return { bg: '#dbeafe', text: '#1e40af' };
    case 'Medium':
      return { bg: '#fef3c7', text: '#92400e' };
    case 'Low':
      return { bg: '#fee2e2', text: '#991b1b' };
    default:
      return { bg: '#f3f4f6', text: '#374151' };
  }
};

const ReliabilityCard = ({ 
  purchaseReliability = 0, 
  consistencyData = null, 
  period = null 
}) => {
  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <Shield size={18} style={{ color: '#2563eb' }} />
        <h3 style={styles.title}>신뢰도 분석</h3>
      </div>
      
      <div style={styles.grid}>
        {/* 구매 신뢰도 */}
        <div style={styles.metric}>
          <div style={styles.metricValue(purchaseReliability)}>
            {purchaseReliability.toFixed(1)}%
          </div>
          <div style={styles.metricLabel}>구매 신뢰도</div>
        </div>
        
        {/* 예측 일관성 */}
        <div style={styles.metric}>
          {consistencyData && consistencyData.consistency_score !== null ? (
            <>
              <div style={styles.metricValue(consistencyData.consistency_score)}>
                {consistencyData.consistency_score.toFixed(1)}
              </div>
              <div style={styles.metricLabel}>예측 일관성</div>
              <div style={{ marginTop: '0.5rem' }}>
                <span style={styles.grade(consistencyData.consistency_grade)}>
                  <Star size={12} />
                  {consistencyData.consistency_grade}
                </span>
              </div>
            </>
          ) : (
            <>
              <div style={styles.metricValue(0)}>-</div>
              <div style={styles.metricLabel}>예측 일관성</div>
              <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#6b7280' }}>
                누적 데이터 필요
              </div>
            </>
          )}
        </div>
      </div>
      
      {/* 추가 정보 */}
      {consistencyData && (
        <div style={{ marginTop: '1rem', fontSize: '0.875rem', color: '#6b7280' }}>
          <p>• 구매 신뢰도: 최적 구간의 점수 비율</p>
          <p>• 예측 일관성: 같은 기간 예측의 변동성 ({consistencyData.prediction_count || 0}회 예측 기준)</p>
        </div>
      )}
    </div>
  );
};

export default ReliabilityCard;