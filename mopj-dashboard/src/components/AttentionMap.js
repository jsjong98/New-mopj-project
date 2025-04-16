import React, { useState } from 'react';
import Modal from 'react-modal';

// Modal 컴포넌트 설정
Modal.setAppElement('#root'); // 접근성을 위해 앱 루트 요소 설정

const styles = {
  container: {
    height: '16rem',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative'
  },
  imageContainer: {
    cursor: 'pointer',
    height: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%'
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
  },
  image: {
    maxWidth: '100%',
    maxHeight: '100%',
    objectFit: 'contain'
  },
  modalImage: {
    width: '100%',
    height: 'auto',
    objectFit: 'contain'
  },
  zoomControls: {
    position: 'absolute',
    bottom: '0.5rem',
    right: '0.5rem',
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    background: 'rgba(255, 255, 255, 0.8)',
    padding: '0.25rem 0.5rem',
    borderRadius: '0.25rem',
    boxShadow: '0 1px 3px rgba(0, 0, 0, 0.1)',
    zIndex: 10
  },
  zoomButton: {
    border: 'none',
    background: '#4b5563',
    color: 'white',
    width: '1.5rem',
    height: '1.5rem',
    borderRadius: '0.25rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '1rem',
    cursor: 'pointer'
  },
  zoomLevel: {
    fontSize: '0.75rem',
    color: '#4b5563',
    minWidth: '2.5rem',
    textAlign: 'center'
  },
  clickHint: {
    position: 'absolute',
    top: '0.5rem',
    left: '0.5rem',
    fontSize: '0.75rem',
    padding: '0.25rem 0.5rem',
    background: 'rgba(0, 0, 0, 0.6)',
    color: 'white',
    borderRadius: '0.25rem',
    zIndex: 10
  },
  modalContent: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    height: '100%'
  },
  modalHeader: {
    width: '100%',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '0.5rem 0',
    marginBottom: '1rem'
  },
  closeButton: {
    border: 'none',
    background: '#ef4444',
    color: 'white',
    padding: '0.5rem 1rem',
    borderRadius: '0.25rem',
    cursor: 'pointer'
  },
  modalZoomControls: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    padding: '0.5rem'
  }
};

const AttentionMap = ({ imageData }) => {
  const [modalIsOpen, setModalIsOpen] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(100);
  const [modalZoomLevel, setModalZoomLevel] = useState(100);

  if (!imageData) {
    return (
      <div style={styles.noDataContainer}>
        <p style={styles.noDataText}>어텐션 맵 데이터가 없습니다</p>
      </div>
    );
  }

  const zoomIn = (setter) => setter(prev => Math.min(prev + 20, 200));
  const zoomOut = (setter) => setter(prev => Math.max(prev - 20, 60));
  const resetZoom = (setter) => setter(100);

  return (
    <div style={styles.container}>
      <div 
        style={styles.imageContainer} 
        onClick={() => setModalIsOpen(true)}
      >
        <div style={styles.clickHint}>클릭하여 크게 보기</div>
        <img 
          src={`data:image/png;base64,${imageData}`} 
          alt="Feature Importance" 
          style={{
            ...styles.image,
            width: `${zoomLevel}%`,
            transformOrigin: 'center center'
          }}
        />
      </div>

      <div style={styles.zoomControls}>
        <button 
          style={styles.zoomButton} 
          onClick={(e) => {
            e.stopPropagation();
            zoomOut(setZoomLevel);
          }}
          title="축소"
        >
          -
        </button>
        <span style={styles.zoomLevel}>{zoomLevel}%</span>
        <button 
          style={styles.zoomButton} 
          onClick={(e) => {
            e.stopPropagation();
            zoomIn(setZoomLevel);
          }}
          title="확대"
        >
          +
        </button>
        <button 
          style={{...styles.zoomButton, width: 'auto', padding: '0 0.5rem'}} 
          onClick={(e) => {
            e.stopPropagation();
            resetZoom(setZoomLevel);
          }}
          title="원본 크기"
        >
          Reset
        </button>
      </div>

      <Modal
        isOpen={modalIsOpen}
        onRequestClose={() => setModalIsOpen(false)}
        style={{
          overlay: {
            backgroundColor: 'rgba(0, 0, 0, 0.75)',
            zIndex: 1000
          },
          content: {
            top: '50%',
            left: '50%',
            right: 'auto',
            bottom: 'auto',
            marginRight: '-50%',
            transform: 'translate(-50%, -50%)',
            width: '90%',
            height: '90%',
            padding: '1rem'
          }
        }}
      >
        <div style={styles.modalContent}>
          <div style={styles.modalHeader}>
            <h2>특성 중요도 시각화 (확대 보기)</h2>
            <div style={styles.modalZoomControls}>
              <button 
                style={styles.zoomButton} 
                onClick={() => zoomOut(setModalZoomLevel)}
                title="축소"
              >
                -
              </button>
              <span style={styles.zoomLevel}>{modalZoomLevel}%</span>
              <button 
                style={styles.zoomButton} 
                onClick={() => zoomIn(setModalZoomLevel)}
                title="확대"
              >
                +
              </button>
              <button 
                style={{...styles.zoomButton, width: 'auto', padding: '0 0.5rem'}} 
                onClick={() => resetZoom(setModalZoomLevel)}
                title="원본 크기"
              >
                Reset
              </button>
              <button 
                style={styles.closeButton} 
                onClick={() => setModalIsOpen(false)}
              >
                닫기
              </button>
            </div>
          </div>
          <div style={{
            overflow: 'auto',
            height: 'calc(100% - 4rem)',
            width: '100%'
          }}>
            <img 
              src={`data:image/png;base64,${imageData}`} 
              alt="Feature Importance (확대)" 
              style={{
                ...styles.modalImage,
                width: `${modalZoomLevel}%`
              }}
            />
          </div>
        </div>
      </Modal>
    </div>
  );
};

export default AttentionMap;