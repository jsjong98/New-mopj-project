import React from 'react';
import { DollarSign } from 'lucide-react';

const styles = {
  container: { height: '100%' },
  noData: {
    height: '16rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#f9fafb',
    borderRadius: '0.375rem'
  },
  noDataText: { color: '#6b7280' },
  tableContainer: { overflowY: 'auto', maxHeight: '16rem' },
  table: { minWidth: '100%', borderCollapse: 'separate', borderSpacing: 0 },
  thead: { backgroundColor: '#f9fafb' },
  th: {
    padding: '0.75rem 1rem',
    textAlign: 'left',
    fontSize: '0.75rem',
    fontWeight: '500',
    color: '#6b7280',
    textTransform: 'uppercase',
    letterSpacing: '0.05em'
  },
  tbody: { backgroundColor: 'white' },
  tr: (isHighlight) => ({
    backgroundColor: isHighlight ? '#f0fdf4' : 'white',
    borderBottom: '1px solid #e5e7eb'
  }),
  td: (isBold) => ({
    padding: '0.5rem 1rem',
    whiteSpace: 'nowrap',
    fontSize: '0.875rem',
    color: '#4b5563',
    fontWeight: isBold ? '500' : 'normal'
  }),

  recommendationBox: {
    marginTop: '1rem',
    padding: '0.75rem',
    border: '1px solid #86efac',
    borderRadius: '0.375rem',
    backgroundColor: '#f0fdf4'
  },
  recommendationTitle: {
    fontWeight: '500',
    color: '#166534',
    marginBottom: '0.5rem',
    display: 'flex',
    alignItems: 'center'
  },
  recommendationIcon: { marginRight: '0.25rem' },
  recommendationDetail: { fontSize: '0.875rem', color: '#166534' },
  boldText: { fontWeight: '500' }
};

const IntervalScoresTable = ({ data, purchaseReliability = 0 }) => {
  // 만약 data가 배열이 아니라면 객체의 값 배열로 변환합니다.
  const scoresData = Array.isArray(data) ? data : Object.values(data || {});
  
  // 필터: 각 아이템이 정의되어 있고 'days' 키가 존재하는지 확인
  const validScores = scoresData.filter((item) => item && typeof item.days !== 'undefined');

  if (validScores.length === 0) {
    return (
      <div style={styles.noData}>
        <p style={styles.noDataText}>구간 점수 데이터가 없습니다</p>
      </div>
    );
  }
  
  // 상위 10개 구간만 표시
  const displayData = validScores.slice(0, 10);
  const bestInterval = displayData[0];



  return (
    <div style={styles.container}>


      <div style={styles.tableContainer}>
        <table style={styles.table}>
          <thead style={styles.thead}>
            <tr>
              <th style={styles.th}>순위</th>
              <th style={styles.th}>시작일</th>
              <th style={styles.th}>종료일</th>
              <th style={styles.th}>일수</th>
              <th style={styles.th}>평균가격</th>
              <th style={styles.th}>점수</th>
            </tr>
          </thead>
          <tbody style={styles.tbody}>
            {displayData.map((interval, idx) => (
              <tr key={idx} style={styles.tr(idx === 0)}>
                <td style={styles.td(true)}>{idx + 1}</td>
                <td style={styles.td(false)}>{interval.start_date}</td>
                <td style={styles.td(false)}>{interval.end_date}</td>
                <td style={styles.td(false)}>{interval.days}</td>
                <td style={styles.td(false)}>{parseFloat(interval.avg_price).toFixed(2)}</td>
                <td style={styles.td(true)}>{interval.score}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {bestInterval && (
        <div style={styles.recommendationBox}>
          <h3 style={styles.recommendationTitle}>
            <DollarSign size={16} style={styles.recommendationIcon} />
            최적 구매 추천
          </h3>
          <div style={styles.recommendationDetail}>
            <p><span style={styles.boldText}>추천 구간:</span> {bestInterval.start_date} ~ {bestInterval.end_date} ({bestInterval.days}일)</p>
            <p><span style={styles.boldText}>예상 평균가:</span> {parseFloat(bestInterval.avg_price).toFixed(2)}</p>
            <p><span style={styles.boldText}>구간 점수:</span> {bestInterval.score} (최고점)</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default IntervalScoresTable;