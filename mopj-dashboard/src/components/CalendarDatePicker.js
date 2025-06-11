import React, { useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Calendar } from 'lucide-react';

// 달력 스타일
const calendarStyles = {
  container: {
    border: '1px solid #e5e7eb',
    borderRadius: '0.5rem',
    backgroundColor: 'white',
    padding: '1rem',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)',
    maxWidth: '320px',
    width: '100%'
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '1rem'
  },
  monthYear: {
    fontSize: '1.125rem',
    fontWeight: '600',
    color: '#374151'
  },
  navButton: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    width: '2rem',
    height: '2rem',
    borderRadius: '0.375rem',
    border: '1px solid #d1d5db',
    backgroundColor: 'white',
    cursor: 'pointer',
    transition: 'all 0.2s'
  },
  weekDays: {
    display: 'grid',
    gridTemplateColumns: 'repeat(7, 1fr)',
    gap: '0.25rem',
    marginBottom: '0.5rem'
  },
  weekDay: {
    padding: '0.5rem',
    textAlign: 'center',
    fontSize: '0.875rem',
    fontWeight: '500',
    color: '#6b7280'
  },
  daysGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(7, 1fr)',
    gap: '0.25rem'
  },
  dayCell: {
    position: 'relative',
    height: '2.5rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: '0.375rem',
    cursor: 'pointer',
    fontSize: '0.875rem',
    transition: 'all 0.2s'
  },
  selectedDay: {
    backgroundColor: '#2563eb',
    color: 'white',
    fontWeight: '600'
  },
  availableDay: {
    backgroundColor: '#f0f9ff',
    color: '#1e40af',
    border: '1px solid #bae6fd'
  },
  unavailableDay: {
    color: '#d1d5db',
    cursor: 'not-allowed'
  },
  otherMonthDay: {
    color: '#e5e7eb',
    cursor: 'not-allowed'
  },
  holidayDay: {
    backgroundColor: '#fef3c7',
    color: '#d97706',
    border: '1px solid #fbbf24'
  },
  autoDetectedHoliday: {
    backgroundColor: '#f0fdf4',
    color: '#16a34a',
    border: '1px solid #4ade80'
  },
  semimonthlyStart: {
    backgroundColor: '#fef3c7',
    color: '#f59e0b',
    border: '2px solid #f59e0b',
    fontWeight: '600'
  },
  badge: {
    position: 'absolute',
    bottom: '0.125rem',
    right: '0.125rem',
    width: '0.5rem',
    height: '0.5rem',
    borderRadius: '50%',
    backgroundColor: '#10b981'
  },
  legend: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '1rem',
    marginTop: '1rem',
    padding: '0.75rem',
    backgroundColor: '#f9fafb',
    borderRadius: '0.375rem',
    fontSize: '0.75rem'
  },
  legendItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '0.25rem'
  },
  legendColor: {
    width: '0.75rem',
    height: '0.75rem',
    borderRadius: '0.125rem'
  }
};

const CalendarDatePicker = ({ 
  availableDates = [], 
  selectedDate, 
  onDateSelect, 
  title = "날짜 선택",
  holidays = []
}) => {
  const [currentMonth, setCurrentMonth] = useState(new Date());
  const [isOpen, setIsOpen] = useState(false);

  // 선택된 날짜가 있으면 해당 월로 이동, 없으면 가장 최근 월로 이동
  useEffect(() => {
    if (selectedDate) {
      // 타임존 이슈 방지를 위해 로컬 날짜로 파싱
      const [year, month, day] = selectedDate.split('-').map(Number);
      const selectedDateObj = new Date(year, month - 1, day);
      setCurrentMonth(selectedDateObj);
    } else if (availableDates.length > 0) {
      // 🎯 선택된 날짜가 없으면 가장 최근 날짜의 월로 이동
      // availableDates를 정렬해서 가장 최근 날짜 찾기
      const sortedDates = [...availableDates].sort((a, b) => {
        const dateA = typeof a === 'string' ? a : a.startDate || a.date;
        const dateB = typeof b === 'string' ? b : b.startDate || b.date;
        return dateB.localeCompare(dateA); // 내림차순 정렬 (최신이 첫 번째)
      });
      
      const latestDate = sortedDates[0];
      const dateKey = typeof latestDate === 'string' ? latestDate : latestDate.startDate || latestDate.date;
      const [year, month] = dateKey.split('-').map(Number);
      const latestDateObj = new Date(year, month - 1, 1);
      setCurrentMonth(latestDateObj);
      
      console.log(`📅 [CALENDAR] Setting initial month to latest available date: ${dateKey} (${year}-${month})`);
    }
  }, [selectedDate, availableDates]);

  // 현재 월의 첫째 날과 마지막 날
  const firstDayOfMonth = new Date(currentMonth.getFullYear(), currentMonth.getMonth(), 1);
  const lastDayOfMonth = new Date(currentMonth.getFullYear(), currentMonth.getMonth() + 1, 0);
  
  // 달력 시작일 (이전 월의 날짜 포함)
  const startDate = new Date(firstDayOfMonth);
  startDate.setDate(startDate.getDate() - firstDayOfMonth.getDay());
  
  // 달력 종료일 (다음 월의 날짜 포함)
  const endDate = new Date(lastDayOfMonth);
  endDate.setDate(endDate.getDate() + (6 - lastDayOfMonth.getDay()));

  // 사용 가능한 날짜를 Set으로 변환 (빠른 검색)
  const availableDateSet = new Set(
    availableDates.map(item => 
      typeof item === 'string' ? item : item.startDate || item.date
    )
  );

  // 휴일을 Set으로 변환 (빠른 검색)
  const holidaySet = new Set(
    holidays.map(holiday => holiday.date)
  );

  // 휴일 체크 함수
  const isHoliday = (dateKey) => {
    return holidays.find(holiday => holiday.date === dateKey);
  };
  
  const getHolidayType = (dateKey) => {
    const holiday = holidays.find(h => h.date === dateKey);
    return holiday ? holiday.source : null;
  };

  // 반월 시작일 체크 함수
  const isSemimonthlyStart = (date) => {
    const day = date.getDate();
    return day === 1 || day === 16;
  };

  // 예측 가능한 날짜에서 반월 시작일인지 체크
  const isSemimonthlyStartDate = (dateKey) => {
    const dateInfo = availableDates.find(item => {
      const itemDateKey = typeof item === 'string' ? item : item.startDate || item.date;
      return itemDateKey === dateKey;
    });
    return dateInfo && dateInfo.isSemimonthlyStart;
  };

  // 달력에 표시할 모든 날짜 생성
  const generateCalendarDays = () => {
    const days = [];
    const current = new Date(startDate);
    
    while (current <= endDate) {
      days.push(new Date(current));
      current.setDate(current.getDate() + 1);
    }
    
    return days;
  };

  const calendarDays = generateCalendarDays();

  // 날짜 포맷팅 (타임존 이슈 방지)
  const formatDateKey = (date) => {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  };

  const formatDisplayDate = (date) => {
    // 타임존 이슈 방지를 위해 로컬 날짜로 처리
    if (typeof date === 'string') {
      const [year, month, day] = date.split('-').map(Number);
      const localDate = new Date(year, month - 1, day);
      return localDate.toLocaleDateString('ko-KR', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        weekday: 'long'
      });
    } else {
      return date.toLocaleDateString('ko-KR', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        weekday: 'long'
      });
    }
  };

  // 월 이동
  const navigateMonth = (direction) => {
    setCurrentMonth(prev => {
      const newDate = new Date(prev);
      newDate.setMonth(prev.getMonth() + direction);
      return newDate;
    });
  };

  // 날짜 선택 처리
  const handleDateClick = (date) => {
    const dateKey = formatDateKey(date);
    const isCurrentMonth = date.getMonth() === currentMonth.getMonth();
    const isAvailable = availableDateSet.has(dateKey);
    
    if (isCurrentMonth && isAvailable) {
      onDateSelect(dateKey);
      setIsOpen(false);
    }
  };

  // 날짜 셀 스타일 계산
  const getDayStyle = (date) => {
    const dateKey = formatDateKey(date);
    const isCurrentMonth = date.getMonth() === currentMonth.getMonth();
    const isAvailable = availableDateSet.has(dateKey);
    const isSelected = selectedDate === dateKey;
    const holidayInfo = isHoliday(dateKey);
    const holidayType = getHolidayType(dateKey);
    const isSemimonthlyStartAvailable = isSemimonthlyStartDate(dateKey);
    
    let style = { ...calendarStyles.dayCell };
    
    if (isSelected) {
      style = { ...style, ...calendarStyles.selectedDay };
    } else if (!isCurrentMonth) {
      style = { ...style, ...calendarStyles.otherMonthDay };
    } else if (holidayInfo && !isAvailable) {
      // 휴일이면서 예측 불가능한 날짜 - 타입에 따라 스타일 구분
      if (holidayType === 'auto_detected') {
        style = { ...style, ...calendarStyles.autoDetectedHoliday };
      } else {
        style = { ...style, ...calendarStyles.holidayDay };
      }
    } else if (isAvailable) {
      // 예측 가능한 날짜 중에서 반월 시작일은 특별히 표시
      if (isSemimonthlyStartAvailable) {
        style = { ...style, ...calendarStyles.semimonthlyStart };
      } else {
        style = { ...style, ...calendarStyles.availableDay };
      }
    } else {
      style = { ...style, ...calendarStyles.unavailableDay };
    }
    
    return style;
  };

  // 선택된 날짜 정보 찾기
  const selectedDateInfo = availableDates.find(item => {
    const dateKey = typeof item === 'string' ? item : item.startDate || item.date;
    return dateKey === selectedDate;
  });

  const weekDays = ['일', '월', '화', '수', '목', '금', '토'];

  return (
    <div style={{ position: 'relative' }}>
      {/* 날짜 선택 버튼 */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '0.5rem',
        padding: '0.75rem',
        border: '1px solid #d1d5db',
        borderRadius: '0.375rem',
        backgroundColor: 'white',
        cursor: 'pointer',
        minHeight: '2.5rem',
        width: '100%'
      }}
      onClick={() => setIsOpen(!isOpen)}>
        <Calendar size={18} style={{ color: '#6b7280' }} />
        <div style={{ flex: 1 }}>
          {selectedDate ? (
            <div>
              <div style={{ fontSize: '0.875rem', fontWeight: '500' }}>
                {formatDisplayDate(selectedDate)}
              </div>
              {selectedDateInfo && selectedDateInfo.label && (
                <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>
                  {selectedDateInfo.label}
                </div>
              )}
            </div>
          ) : (
            <span style={{ color: '#9ca3af' }}>{title}</span>
          )}
        </div>
        <ChevronRight 
          size={16} 
          style={{ 
            color: '#6b7280',
            transform: isOpen ? 'rotate(90deg)' : 'rotate(0deg)',
            transition: 'transform 0.2s'
          }} 
        />
      </div>

      {/* 달력 패널 */}
      {isOpen && (
        <div style={{
          position: 'absolute',
          top: '100%',
          left: 0,
          zIndex: 1000,
          marginTop: '0.25rem',
          ...calendarStyles.container
        }}>
          {/* 달력 헤더 */}
          <div style={calendarStyles.header}>
            <button
              style={calendarStyles.navButton}
              onClick={() => navigateMonth(-1)}
              onMouseEnter={(e) => e.target.style.backgroundColor = '#f3f4f6'}
              onMouseLeave={(e) => e.target.style.backgroundColor = 'white'}
            >
              <ChevronLeft size={16} />
            </button>
            
            <div style={calendarStyles.monthYear}>
              {currentMonth.toLocaleDateString('ko-KR', { 
                year: 'numeric', 
                month: 'long' 
              })}
            </div>
            
            <button
              style={calendarStyles.navButton}
              onClick={() => navigateMonth(1)}
              onMouseEnter={(e) => e.target.style.backgroundColor = '#f3f4f6'}
              onMouseLeave={(e) => e.target.style.backgroundColor = 'white'}
            >
              <ChevronRight size={16} />
            </button>
          </div>

          {/* 요일 헤더 */}
          <div style={calendarStyles.weekDays}>
            {weekDays.map(day => (
              <div key={day} style={calendarStyles.weekDay}>
                {day}
              </div>
            ))}
          </div>

          {/* 날짜 그리드 */}
          <div style={calendarStyles.daysGrid}>
            {calendarDays.map((date, index) => {
              const dateKey = formatDateKey(date);
              const isAvailable = availableDateSet.has(dateKey);
              const isCurrentMonth = date.getMonth() === currentMonth.getMonth();
              
              return (
                <div
                  key={index}
                  style={getDayStyle(date)}
                  onClick={() => handleDateClick(date)}
                  onMouseEnter={(e) => {
                    if (isCurrentMonth && isAvailable && selectedDate !== dateKey) {
                      e.target.style.backgroundColor = '#dbeafe';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (isCurrentMonth && isAvailable && selectedDate !== dateKey) {
                      e.target.style.backgroundColor = '#f0f9ff';
                    }
                  }}
                >
                  {date.getDate()}
                  {isAvailable && isCurrentMonth && (
                    <div style={calendarStyles.badge}></div>
                  )}
                </div>
              );
            })}
          </div>

          {/* 범례 */}
          <div style={calendarStyles.legend}>
            <div style={calendarStyles.legendItem}>
              <div style={{
                ...calendarStyles.legendColor,
                backgroundColor: '#2563eb'
              }}></div>
              <span>선택됨</span>
            </div>
            <div style={calendarStyles.legendItem}>
              <div style={{
                ...calendarStyles.legendColor,
                backgroundColor: '#bae6fd'
              }}></div>
              <span>예측 가능</span>
            </div>
            <div style={calendarStyles.legendItem}>
              <div style={{
                ...calendarStyles.legendColor,
                backgroundColor: '#f59e0b',
                border: '1px solid #f59e0b'
              }}></div>
              <span>반월 시작일</span>
            </div>
            <div style={calendarStyles.legendItem}>
              <div style={{
                ...calendarStyles.legendColor,
                backgroundColor: '#10b981'
              }}></div>
              <span>가능일 표시</span>
            </div>
            {holidays.some(h => h.source === 'file') && (
              <div style={calendarStyles.legendItem}>
                <div style={{
                  ...calendarStyles.legendColor,
                  backgroundColor: '#fbbf24'
                }}></div>
                <span>휴일 (파일)</span>
              </div>
            )}
            {holidays.some(h => h.source === 'auto_detected') && (
              <div style={calendarStyles.legendItem}>
                <div style={{
                  ...calendarStyles.legendColor,
                  backgroundColor: '#4ade80'
                }}></div>
                <span>휴일 (자동감지)</span>
              </div>
            )}
          </div>

          {/* 닫기 버튼 */}
          <div style={{ textAlign: 'center', marginTop: '0.75rem' }}>
            <button
              style={{
                padding: '0.375rem 0.75rem',
                fontSize: '0.75rem',
                color: '#6b7280',
                backgroundColor: '#f9fafb',
                border: '1px solid #e5e7eb',
                borderRadius: '0.375rem',
                cursor: 'pointer'
              }}
              onClick={() => setIsOpen(false)}
            >
              닫기
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default CalendarDatePicker;
