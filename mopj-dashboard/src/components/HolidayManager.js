import React, { useState, useEffect } from 'react';
import { getHolidays, uploadHolidayFile, reloadHolidays } from '../services/api';
import FileUploader from './FileUploader';

const HolidayManager = () => {
  const [holidays, setHolidays] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  useEffect(() => {
    fetchHolidays();
  }, []);

  const fetchHolidays = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await getHolidays();
      if (response.success) {
        setHolidays(response.holidays || []);
        setSuccessMessage(`${response.count || 0}개의 휴일 정보가 로드되었습니다.`);
      } else {
        setError(response.error || '휴일 정보를 불러오는데 실패했습니다.');
      }
    } catch (err) {
      console.error('Error fetching holidays:', err);
      setError(err.message || '휴일 정보를 불러오는데 실패했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleUploadSuccess = async (data) => {
    try {
      setIsLoading(true);
      setError(null);
      setSuccessMessage(null);

      // 파일 업로드 처리
      const response = await uploadHolidayFile(data.file);
      
      if (response.success) {
        setHolidays(response.holidays || []);
        setSuccessMessage('휴일 파일이 성공적으로 업로드되었습니다.');
      } else {
        setError(response.error || '휴일 파일 업로드에 실패했습니다.');
      }
    } catch (err) {
      console.error('Error uploading holiday file:', err);
      setError(err.message || '휴일 파일 업로드 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReload = async () => {
    try {
      setIsLoading(true);
      setError(null);
      setSuccessMessage(null);

      const response = await reloadHolidays();
      
      if (response.success) {
        setHolidays(response.holidays || []);
        setSuccessMessage('휴일 정보가 성공적으로 새로고침되었습니다.');
      } else {
        setError(response.error || '휴일 정보 새로고침에 실패했습니다.');
      }
    } catch (err) {
      console.error('Error reloading holidays:', err);
      setError(err.message || '휴일 정보 새로고침 중 오류가 발생했습니다.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="holiday-manager">
      <h2 className="section-title">휴일 관리</h2>
      
      <div className="card">
        <div className="card-body">
          <p className="card-description">
            휴일 정보는 CSV 또는 Excel 파일로 관리됩니다. 휴일은 예측 시 영업일에서 제외됩니다.
            파일의 첫 번째 열은 'date' 컬럼이어야 하며, YYYY-MM-DD 형식의 휴일 날짜를 포함해야 합니다.
          </p>
          
          <div className="holidays-container">
            <h3>현재 등록된 휴일 목록 ({holidays.length}개)</h3>
            {holidays.length === 0 ? (
              <p className="no-data-message">현재 등록된 휴일이 없습니다.</p>
            ) : (
              <div className="holiday-list">
                {holidays.map((holiday, index) => (
                  <div key={index} className="holiday-item">
                    {typeof holiday === 'object' ? (
                      <div>
                        <div className="holiday-date">{holiday.date}</div>
                        {holiday.description && (
                          <div className="holiday-description">{holiday.description}</div>
                        )}
                        {holiday.source && (
                          <div className="holiday-source">출처: {holiday.source}</div>
                        )}
                      </div>
                    ) : (
                      <div className="holiday-date">{holiday}</div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
          
          <div className="upload-section">
            <h3>휴일 파일 업로드</h3>
            <FileUploader
              onUploadSuccess={handleUploadSuccess}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
              acceptedFormats=".csv,.xlsx"
              fileType="휴일"
            />
          </div>
          
          <div className="actions">
            <button 
              className="btn btn-primary"
              onClick={handleReload}
              disabled={isLoading}
            >
              {isLoading ? '로딩 중...' : '휴일 정보 새로고침'}
            </button>
          </div>
          
          {error && (
            <div className="alert alert-danger">
              <strong>오류:</strong> {error}
            </div>
          )}
          
          {successMessage && (
            <div className="alert alert-success">
              <strong>성공:</strong> {successMessage}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default HolidayManager;
