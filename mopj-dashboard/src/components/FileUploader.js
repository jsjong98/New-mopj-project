import React, { useState } from 'react';
import { Upload, AlertTriangle } from 'lucide-react';
import { uploadCSV, getAvailableDates, getFileMetadata } from '../services/api';

const styles = {
  container: (dragActive) => ({
    border: '2px dashed',
    borderColor: dragActive ? '#3b82f6' : '#d1d5db',
    borderRadius: '0.5rem',
    padding: '1.5rem',
    textAlign: 'center',
    transition: 'colors 0.3s',
    backgroundColor: dragActive ? '#eff6ff' : 'transparent'
  }),
  icon: {
    height: '3rem',
    width: '3rem',
    color: '#9ca3af',
    margin: '0 auto 0.5rem auto'
  },
  text: {
    color: '#4b5563',
    marginBottom: '0.5rem'
  },
  smallText: {
    fontSize: '0.875rem',
    color: '#6b7280'
  },
  hidden: {
    display: 'none'
  },
  button: (isLoading) => ({
    marginTop: '1rem',
    display: 'inline-block',
    backgroundColor: '#2563eb',
    color: 'white',
    padding: '0.5rem 1rem',
    borderRadius: '0.375rem',
    transition: 'background-color 0.3s',
    cursor: isLoading ? 'not-allowed' : 'pointer',
    opacity: isLoading ? 0.5 : 1,
  }),
  errorContainer: {
    marginTop: '0.75rem',
    color: '#ef4444',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center'
  },
  errorIcon: {
    marginRight: '0.25rem'
  }
};

const FileUploader = ({ onUploadSuccess, isLoading, setIsLoading }) => {
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      await handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = async (e) => {
    if (e.target.files && e.target.files[0]) {
      await handleFileUpload(e.target.files[0]);
    }
  };

  const handleFileUpload = async (file) => {
    // 파일 형식 및 크기 확인 코드는 유지
    
    setIsLoading(true);
    setError(null);
    
    try {
      // 파일 업로드
      const uploadResult = await uploadCSV(file);
      
      // 오류 확인
      if (uploadResult.error) {
        setError(uploadResult.error);
        return;
      }
      
      // 업로드 성공 후 날짜 정보 요청
      const datesResult = await getAvailableDates(uploadResult.filepath);
      
      // 날짜 정보 오류 확인
      if (datesResult.error) {
        setError(datesResult.error);
        return;
      }
      
      // 성공 콜백 호출
      onUploadSuccess({
        filepath: uploadResult.filepath,
        dates: datesResult.dates || [],
        latestDate: datesResult.latest_date
      });
    } catch (err) {
      console.error('File upload failed:', err);
      setError(err.error || err.message || '파일 업로드 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };
  
  return (
    <div 
      style={styles.container(dragActive)}
      onDragEnter={handleDrag}
      onDragOver={handleDrag}
      onDragLeave={handleDrag}
      onDrop={handleDrop}
    >
      <Upload style={styles.icon} />
      <p style={styles.text}>CSV 파일을 드래그하여 업로드하거나 클릭하여 선택하세요</p>
      <p style={styles.smallText}>지원 형식: .csv</p>
      
      <input
        type="file"
        id="file-upload"
        style={styles.hidden}
        accept=".csv"
        onChange={handleFileChange}
        disabled={isLoading}
      />
      
      <label 
        htmlFor="file-upload"
        style={styles.button(isLoading)}
      >
        {isLoading ? '처리 중...' : 'CSV 파일 업로드'}
      </label>
      
      {error && (
        <div style={styles.errorContainer}>
          <AlertTriangle size={16} style={styles.errorIcon} />
          {error}
        </div>
      )}
    </div>
  );
};

export default FileUploader;