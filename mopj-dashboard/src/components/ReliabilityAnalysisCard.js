// 새 컴포넌트: ReliabilityAnalysisCard.js
import React from 'react';
import { Shield, CheckCircle, AlertTriangle, XCircle } from 'lucide-react';

const styles = {
  card: {
    backgroundColor: 'white',
    borderRadius: '0.5rem',
    padding: '1.5rem',
    marginBottom: '1.5rem',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    marginBottom: '1.5rem'
  },
  title: {
    fontSize: '1.25rem',
    fontWeight: '600',
    color: '#374151',
    marginLeft: '0.5rem'
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '1rem',
    marginBottom: '1.5rem'
  },
  metric: {
    textAlign: 'center',
    padding: '1rem',
    backgroundColor: '#f9fafb',
    borderRadius: '0.5rem'
  },
  metricValue: (score, threshold) => ({
    fontSize: '2rem',
    fontWeight: 'bold',
    color: score >= threshold ? '#10b981' : '#ef4444'
  }),
  metricLabel: {
    fontSize: '0.875rem',
    color: '#6b7280',
    marginTop: '0.5rem'
  },
  threshold: {
    fontSize: '0.75rem',
    color: '#6b7280',
    marginTop: '0.25rem'
  },
  judgmentBox: (level) => ({
    padding: '1rem',
    borderRadius: '0.5rem',
    border: `2px solid ${getJudgmentColor(level).border}`,
    backgroundColor: getJudgmentColor(level).bg
  }),
  judgmentHeader: (level) => ({
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    marginBottom: '0.75rem',
    fontSize: '1.125rem',
    fontWeight: '600',
    color: getJudgmentColor(level).text
  }),
  judgmentText: (level) => ({
    color: getJudgmentColor(level).text,
    fontSize: '0.9rem',
    lineHeight: '1.5'
  }),
  detailsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '0.5rem',
    marginTop: '0.75rem',
    fontSize: '0.75rem',
    color: '#6b7280'
  }
};

const getJudgmentColor = (level) => {
  switch (level) {
    case 'excellent':
      return { bg: '#d1fae5', border: '#10b981', text: '#065f46' };
    case 'caution':
      return { bg: '#fef3c7', border: '#f59e0b', text: '#92400e' };
    case 'reject':
      return { bg: '#fee2e2', border: '#ef4444', text: '#991b1b' };
    default:
      return { bg: '#f3f4f6', border: '#9ca3af', text: '#374151' };
  }
};

const getJudgmentIcon = (level) => {
  switch (level) {
    case 'excellent':
      return <CheckCircle size={20} />;
    case 'caution':
      return <AlertTriangle size={20} />;
    case 'reject':
      return <XCircle size={20} />;
    default:
      return <Shield size={20} />;
  }
};

const ReliabilityAnalysisCard = ({ 
  consistencyScores = null, 
  purchaseReliability = 0,
  actualBusinessDays = 0
}) => {
  // 신뢰도 점수 계산
  const predictionReliability = consistencyScores && consistencyScores.consistency_score 
    ? consistencyScores.consistency_score 
    : 0;

  // 3단계 판정 로직
  const getJudgment = () => {
    if (predictionReliability < 96) {
      return {
        level: 'reject',
        title: '예측 결과 활용 불가',
        message: '예측 신뢰도가 기준(96점) 미만입니다. 예측 결과를 신뢰하기 어려우므로 활용을 권장하지 않습니다.',
        recommendation: '더 많은 데이터를 축적하거나 모델 파라미터를 재조정해보세요.'
      };
    } else if (purchaseReliability < 63.7) {
      return {
        level: 'caution',
        title: '예측 활용, 구매 기간 주의',
        message: '예측 신뢰도는 높으나 구매 신뢰도가 기준(63.7%) 미만입니다. 가격 예측은 참고하되 특정 구매 기간 추천은 신중히 검토하세요.',
        recommendation: '예측된 가격 추이는 활용하되, 구매 타이밍은 추가적인 시장 분석과 함께 결정하세요.'
      };
    } else {
      return {
        level: 'excellent',
        title: '예측 및 구매 전략 활용 권장',
        message: '예측 신뢰도와 구매 신뢰도 모두 기준을 통과했습니다. 예측 결과와 추천 구매 기간을 안심하고 활용하세요.',
        recommendation: '제시된 최적 구매 구간을 적극 활용하여 구매 전략을 수립하세요.'
      };
    }
  };

  const judgment = getJudgment();
  const maxScore = actualBusinessDays * 3;

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <Shield size={24} style={{ color: '#2563eb' }} />
        <h3 style={styles.title}>신뢰도 종합 분석</h3>
      </div>
      
      {/* 신뢰도 지표 */}
      <div style={styles.metricsGrid}>
        <div style={styles.metric}>
          <div style={styles.metricValue(predictionReliability, 96)}>
            {predictionReliability.toFixed(1)}
          </div>
          <div style={styles.metricLabel}>예측 신뢰도</div>
          <div style={styles.threshold}>기준: 96점 이상</div>
        </div>
        
        <div style={styles.metric}>
          <div style={styles.metricValue(purchaseReliability, 63.7)}>
            {purchaseReliability.toFixed(1)}%
          </div>
          <div style={styles.metricLabel}>구매 신뢰도</div>
          <div style={styles.threshold}>기준: 63.7% 이상</div>
        </div>
      </div>
      
      {/* 종합 판정 */}
      <div style={styles.judgmentBox(judgment.level)}>
        <div style={styles.judgmentHeader(judgment.level)}>
          {getJudgmentIcon(judgment.level)}
          {judgment.title}
        </div>
        <div style={styles.judgmentText(judgment.level)}>
          <p>{judgment.message}</p>
          <p style={{ marginTop: '0.5rem', fontWeight: '500' }}>
            💡 {judgment.recommendation}
          </p>
        </div>
        
        {/* 상세 정보 */}
        <div style={styles.detailsGrid}>
          <div>예측 기간: {actualBusinessDays}일</div>
          <div>최대 가능 점수: {maxScore}점</div>
          <div>예측 횟수: {consistencyScores ? consistencyScores.prediction_count : 0}회</div>
          <div>신뢰도 등급: {consistencyScores ? consistencyScores.consistency_grade : 'N/A'}</div>
        </div>
      </div>
    </div>
  );
};

export default ReliabilityAnalysisCard;