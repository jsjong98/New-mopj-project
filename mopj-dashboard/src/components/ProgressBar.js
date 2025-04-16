import React from 'react';

const styles = {
  progressBarOuter: {
    width: '100%',
    backgroundColor: '#e5e7eb',
    borderRadius: '9999px',
    height: '1rem',
    marginBottom: '1rem'
  },
  progressBarInner: (progress) => ({
    backgroundColor: '#2563eb',
    height: '1rem',
    borderRadius: '9999px',
    transition: 'width 0.3s ease-in-out',
    width: `${progress}%`
  })
};

const ProgressBar = ({ progress }) => {
  return (
    <div style={styles.progressBarOuter}>
      <div style={styles.progressBarInner(progress)} />
    </div>
  );
};

export default ProgressBar;