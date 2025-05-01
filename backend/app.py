from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
import os
import json
import warnings
import random
import traceback
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')  # 서버에서 GUI 백엔드 사용 안 함
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns
from werkzeug.utils import secure_filename
import io
import base64
import tempfile
import time
from threading import Thread
import logging
import calendar
import shutil
import optuna

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# 랜덤 시드 설정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # GPU 사용 시
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 저장할 디렉토리 설정
PLOT_DIR = 'static/plots'
REPORT_DIR = 'static/reports'
MA_PLOT_DIR = 'static/ma_plots'
ATTENTION_DIR = 'static/attention'
UPLOAD_FOLDER = 'uploads'
MODELS_DIR = 'models'

# 디렉토리 생성
for d in [PLOT_DIR, REPORT_DIR, MA_PLOT_DIR, ATTENTION_DIR, UPLOAD_FOLDER, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask 설정
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 최대 파일 크기 32MB로 증가

# 전역 상태 변수
prediction_state = {
    'current_data': None,
    'latest_predictions': None,
    'latest_interval_scores': None,
    'latest_attention_data': None,
    'latest_ma_results': None,
    'current_date': None,
    'is_predicting': False,
    'prediction_progress': 0,
    'error': None,
    'selected_features': None,
    'feature_importance': None,
    'semimonthly_period': None,
    'next_semimonthly_period': None,
    'accumulated_predictions': [],  # 여러 날짜의 예측 결과 저장
    'accumulated_metrics': {},      # 누적 성능 지표
    'prediction_dates': []          # 예측이 수행된 날짜들
}

# 데이터 로더의 워커 시드 고정을 위한 함수
def seed_worker(worker_id):
    worker_seed = SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# 데이터 로더의 생성자 시드 고정
g = torch.Generator()
g.manual_seed(SEED)

#######################################################################
# 모델 및 유틸리티 함수
#######################################################################

# 날짜 포맷팅 유틸리티 함수
def format_date(date_obj, format_str='%Y-%m-%d'):
    """날짜 객체를 문자열로 안전하게 변환"""
    try:
        # pandas Timestamp 또는 datetime.datetime
        if hasattr(date_obj, 'strftime'):
            return date_obj.strftime(format_str)
        
        # numpy.datetime64
        elif isinstance(date_obj, np.datetime64):
            # 날짜 포맷이 'YYYY-MM-DD'인 경우
            return str(date_obj)[:10]
        
        # 문자열인 경우 이미 날짜 형식이라면 추가 처리
        elif isinstance(date_obj, str):
            # GMT 형식이면 파싱하여 변환
            if 'GMT' in date_obj:
                parsed_date = datetime.strptime(date_obj, '%a, %d %b %Y %H:%M:%S GMT')
                return parsed_date.strftime(format_str)
            return date_obj[:10] if len(date_obj) > 10 else date_obj
        
        # 그 외 경우
        else:
            return str(date_obj)
    
    except Exception as e:
        logger.warning(f"날짜 포맷팅 오류: {str(e)}")
        return str(date_obj)

# 데이터 로딩 및 전처리 함수
def load_data(file_path):
    """데이터 로드 및 기본 전처리"""
    logger.info("Loading data...")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    logger.info(f"Original data shape: {df.shape}")
    
    # 모든 inf 값을 NaN으로 변환
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 결측치 처리 - 모든 컬럼에 동일하게 적용
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 처리 후 남아있는 inf나 nan 확인
    if df.isnull().any().any() or np.isinf(df.values).any():
        logger.warning("Dataset still contains NaN or inf values after preprocessing")
        # 문제가 있는 열 출력
        problematic_cols = df.columns[
            df.isnull().any() | np.isinf(df).any()
        ]
        logger.warning(f"Problematic columns: {problematic_cols}")
        
        # 추가적인 전처리: 남은 inf/nan 값을 해당 컬럼의 평균값으로 대체
        for col in problematic_cols:
            col_mean = df[col].replace([np.inf, -np.inf], np.nan).mean()
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(col_mean)
    
    logger.info(f"Final shape after preprocessing: {df.shape}")
    return df

# 변수 그룹 정의
variable_groups = {
    'crude_oil': ['WTI', 'Brent', 'Dubai'],
    'gasoline': ['Gasoline_92RON', 'Gasoline_95RON', 'Europe_M.G_10ppm', 'RBOB (NYMEX)_M1'],
    'naphtha': ['MOPAG', 'MOPS', 'Europe_CIF NWE'],
    'lpg': ['C3_LPG', 'C4_LPG'],
    'product': ['EL_CRF NEA', 'EL_CRF SEA', 'PL_FOB Korea', 'BZ_FOB Korea', 'BZ_FOB SEA', 'BZ_FOB US M1', 'BZ_FOB US M2', 'TL_FOB Korea', 'TL_FOB US M1', 'TL_FOB US M2',
    'MX_FOB Korea', 'PX_FOB Korea', 'SM_FOB Korea', 'RPG Value_FOB PG', 'FO_HSFO 180 CST', 'MTBE_FOB Singapore'],
    'spread': ['biweekly Spread','BZ_H2-TIME SPREAD', 'Brent_WTI', 'MOPJ_MOPAG', 'MOPJ_MOPS', 'Naphtha_Spread', 'MG92_E Nap', 'C3_MOPJ', 'C4_MOPJ', 'Nap_Dubai',
    'MG92_Nap_MOPS', '95R_92R_Asia', 'M1_M2_RBOB', 'RBOB_Brent_m1', 'RBOB_Brent_m2', 'EL_MOPJ', 'PL_MOPJ', 'BZ_MOPJ', 'TL_MOPJ', 'PX_MOPJ', 'HD_EL', 'LD_EL', 'LLD_EL', 'PP_PL',
    'SM_EL+BZ', 'US_FOBK_BZ', 'NAP_HSFO_180', 'MTBE_MOPJ'],
    'economics': ['Dow_Jones', 'Euro', 'Gold'],
    'freight': ['Freight_55_PG', 'Freight_55_Maili', 'Freight_55_Yosu', 'Freight_55_Daes', 'Freight_55_Chiba',
    'Freight_75_PG', 'Freight_75_Maili', 'Freight_75_Yosu', 'Freight_75_Daes', 'Freight_75_Chiba', 'Flat Rate_PG', 'Flat Rate_Maili', 'Flat Rate_Yosu', 'Flat Rate_Daes',
    'Flat Rate_Chiba'],
    'ETF': ['DIG', 'DUG', 'IYE', 'VDE', 'XLE']
}

def load_holidays_from_file(filepath=None):
    """
    CSV 또는 Excel 파일에서 휴일 목록을 로드하는 함수
    
    Args:
        filepath (str): 휴일 목록 파일 경로, None이면 기본 경로 사용
    
    Returns:
        set: 휴일 날짜 집합 (YYYY-MM-DD 형식)
    """
    # 기본 파일 경로
    if filepath is None:
        filepath = os.path.join('models', 'holidays.csv')
    
    # 파일 확장자 확인
    _, ext = os.path.splitext(filepath)
    
    # 파일이 존재하지 않으면 기본 휴일 목록 생성
    if not os.path.exists(filepath):
        logger.warning(f"Holiday file {filepath} not found. Creating default holiday file.")
        
        # 기본 2025년 싱가폴 공휴일
        default_holidays = [
            "2025-01-01", "2025-01-29", "2025-01-30", "2025-03-31", "2025-04-18", 
            "2025-05-01", "2025-05-12", "2025-06-07", "2025-08-09", "2025-10-20", 
            "2025-12-25", "2026-01-01"
        ]
        
        # 기본 파일 생성
        df = pd.DataFrame({'date': default_holidays, 'description': ['Singapore Holiday']*len(default_holidays)})
        
        if ext.lower() == '.xlsx':
            df.to_excel(filepath, index=False)
        else:
            df.to_csv(filepath, index=False)
        
        logger.info(f"Created default holiday file at {filepath}")
        return set(default_holidays)
    
    try:
        # 파일 로드
        if ext.lower() == '.xlsx':
            df = pd.read_excel(filepath)
        else:
            df = pd.read_csv(filepath)
        
        # 'date' 컬럼이 있는지 확인
        if 'date' not in df.columns:
            logger.error(f"Holiday file {filepath} does not have 'date' column")
            return set()
        
        # 날짜 형식 표준화
        holidays = set()
        for date_str in df['date']:
            try:
                date = pd.to_datetime(date_str)
                holidays.add(date.strftime('%Y-%m-%d'))
            except:
                logger.warning(f"Invalid date format: {date_str}")
        
        logger.info(f"Loaded {len(holidays)} holidays from {filepath}")
        return holidays
        
    except Exception as e:
        logger.error(f"Error loading holiday file: {str(e)}")
        logger.error(traceback.format_exc())
        return set()

# 전역 변수로 휴일 집합 관리
holidays = load_holidays_from_file()

def is_holiday(date):
    """주어진 날짜가 휴일인지 확인하는 함수"""
    date_str = format_date(date, '%Y-%m-%d')
    return date_str in holidays

# 휴일 정보 업데이트 함수
def update_holidays(filepath=None):
    """휴일 정보를 재로드하는 함수"""
    global holidays
    holidays = load_holidays_from_file(filepath)
    return holidays

# TimeSeriesDataset 및 평가 메트릭스
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, device, prev_values=None):
        if isinstance(X, torch.Tensor):
            self.X = X
            self.y = y
        else:
            self.X = torch.FloatTensor(X).to(device)
            self.y = torch.FloatTensor(y).to(device)
        
        if prev_values is not None:
            if isinstance(prev_values, torch.Tensor):
                self.prev_values = prev_values
            else:
                self.prev_values = torch.FloatTensor(prev_values).to(device)
        else:
            self.prev_values = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.prev_values is not None:
            return self.X[idx], self.y[idx], self.prev_values[idx]
        return self.X[idx], self.y[idx]

# 복합 손실 함수
def composite_loss(pred, target, prev_value, alpha=0.6, beta=0.2, gamma=0.15, delta=0.05, eps=1e-8):
    """
    Composite loss for:
      - MSE loss (for overall price accuracy)
      - Volatility loss (to match the magnitude of changes)
      - Directional loss (surrogate for F1-score improvement)
      - Continuity loss (to ensure smooth transition)
    """
    # MSE Loss
    loss_mse = F.mse_loss(pred, target)
    
    # Volatility Loss
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    loss_vol = torch.mean(torch.abs(torch.log1p(torch.abs(pred_diff) + eps) - 
                                  torch.log1p(torch.abs(target_diff) + eps)))
    
    # Directional Loss
    cos_sim = torch.cosine_similarity(pred_diff, target_diff, dim=-1)
    loss_dir = 1 - cos_sim.mean()
    
    # Continuity Loss
    loss_cont = torch.mean(torch.abs(pred[:, 0:1] - prev_value))
    
    total_loss = alpha * loss_mse + beta * loss_vol + gamma * loss_dir + delta * loss_cont
    loss_info = {
        'mse_loss': loss_mse.item(),
        'volatility_loss': loss_vol.item(),
        'directional_loss': loss_dir.item(),
        'continuity_loss': loss_cont.item(),
        'total_loss': total_loss.item()
    }
    return total_loss, loss_info

# 학습률 스케줄러 클래스
class CombinedScheduler:
    def __init__(self, optimizer, warmup_steps, max_lr, factor=0.5, patience=5, min_lr=1e-6):
        self.warmup_scheduler = WarmupScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            max_lr=max_lr
        )
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            verbose=True,
            min_lr=min_lr
        )
        self.current_step = 0
        self.warmup_steps = warmup_steps
    
    def step(self, val_loss=None):
        # Warmup 단계에서는 매 배치마다 호출
        if self.current_step < self.warmup_steps:
            self.warmup_scheduler.step()
            self.current_step += 1
        # Warmup 이후에는 validation loss에 따라 조정
        elif val_loss is not None:
            self.plateau_scheduler.step(val_loss)

class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, max_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        # warmup 단계 동안 선형 증가
        lr = self.max_lr * self.current_step / self.warmup_steps
        # warmup 단계를 초과하면 max_lr로 고정
        if self.current_step > self.warmup_steps:
            lr = self.max_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# 개선된 LSTM 예측 모델
class ImprovedLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2, output_size=23):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # hidden_size를 8의 배수로 조정
        self.adjusted_hidden = (hidden_size // 8) * 8
        if self.adjusted_hidden < 32:
            self.adjusted_hidden = 32
        
        # LSTM dropout 설정
        self.lstm_dropout = 0.0 if num_layers == 1 else dropout
        
        # 계층적 LSTM 구조
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=input_size if i == 0 else self.adjusted_hidden,
                hidden_size=self.adjusted_hidden,
                num_layers=1,
                batch_first=True
            ) for i in range(num_layers)
        ])
        
        # 듀얼 어텐션 메커니즘
        self.temporal_attention = nn.MultiheadAttention(
            self.adjusted_hidden,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        self.feature_attention = nn.MultiheadAttention(
            self.adjusted_hidden,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.adjusted_hidden) for _ in range(num_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(self.adjusted_hidden)
        
        # Dropout 레이어
        self.dropout_layer = nn.Dropout(dropout)
        
        # 이전 값 정보를 결합하기 위한 레이어
        self.prev_value_encoder = nn.Sequential(
            nn.Linear(1, self.adjusted_hidden // 4),
            nn.ReLU(),
            nn.Linear(self.adjusted_hidden // 4, self.adjusted_hidden)
        )
        
        # 시계열 특성 추출을 위한 컨볼루션 레이어
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.adjusted_hidden, self.adjusted_hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.adjusted_hidden, self.adjusted_hidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 출력 레이어 - 계층적 구조
        self.output_layers = nn.ModuleList([
            nn.Linear(self.adjusted_hidden, self.adjusted_hidden // 2),
            nn.Linear(self.adjusted_hidden // 2, self.adjusted_hidden // 4),
            nn.Linear(self.adjusted_hidden // 4, output_size)
        ])
        
        # 잔차 연결을 위한 프로젝션 레이어
        self.residual_proj = nn.Linear(self.adjusted_hidden, output_size)
        
    def forward(self, x, prev_value=None, return_attention=False):
        batch_size = x.size(0)
        
        # 계층적 LSTM 처리
        lstm_out = x
        skip_connections = []
        
        for i, (lstm, layer_norm) in enumerate(zip(self.lstm_layers, self.layer_norms)):
            lstm_out, _ = lstm(lstm_out)
            lstm_out = layer_norm(lstm_out)
            lstm_out = self.dropout_layer(lstm_out)
            skip_connections.append(lstm_out)
        
        # 시간적 어텐션
        temporal_context, temporal_weights = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        temporal_context = self.dropout_layer(temporal_context)
        
        # 특징 어텐션
        # 특징 차원으로 변환 (B, L, H) -> (B, H, L)
        feature_input = lstm_out.transpose(1, 2)
        feature_input = self.conv_layers(feature_input)
        feature_input = feature_input.transpose(1, 2)
        
        feature_context, feature_weights = self.feature_attention(feature_input, feature_input, feature_input)
        feature_context = self.dropout_layer(feature_context)
        
        # 컨텍스트 결합
        combined_context = temporal_context + feature_context
        for skip in skip_connections:
            combined_context = combined_context + skip
        
        combined_context = self.final_layer_norm(combined_context)
        
        # 이전 값 정보 처리
        if prev_value is not None:
            prev_value = prev_value.unsqueeze(1) if len(prev_value.shape) == 1 else prev_value
            prev_encoded = self.prev_value_encoder(prev_value)
            combined_context = combined_context + prev_encoded.unsqueeze(1)
        
        # 최종 특징 추출 (마지막 시퀀스)
        final_features = combined_context[:, -1, :]
        
        # 계층적 출력 처리
        out = final_features
        residual = self.residual_proj(final_features)
        
        for i, layer in enumerate(self.output_layers):
            out = layer(out)
            if i < len(self.output_layers) - 1:
                out = F.relu(out)
                out = self.dropout_layer(out)
        
        # 잔차 연결 추가
        out = out + residual
        
        if return_attention:
            attention_weights = {
                'temporal_weights': temporal_weights,
                'feature_weights': feature_weights
            }
            return out, attention_weights
        
        return out
        
    def get_attention_maps(self, x, prev_value=None):
        """어텐션 가중치 맵을 반환하는 함수"""
        with torch.no_grad():
            # forward 메서드에 return_attention=True 전달
            _, attention_weights = self.forward(x, prev_value, return_attention=True)
            
            # 어텐션 가중치 평균 계산 (multi-head -> single map)
            temporal_weights = attention_weights['temporal_weights'].mean(dim=1)  # 헤드 평균
            feature_weights = attention_weights['feature_weights'].mean(dim=1)    # 헤드 평균
            
            return {
                'temporal_weights': temporal_weights.cpu().numpy(),
                'feature_weights': feature_weights.cpu().numpy()
            }

#######################################################################
# 반월 기간 관련 함수
#######################################################################

# 1. 반월 기간 계산 함수
def get_semimonthly_period(date):
    """
    날짜를 반월 기간으로 변환하는 함수
    - 1일~15일: "YYYY-MM-SM1"
    - 16일~말일: "YYYY-MM-SM2"
    """
    year = date.year
    month = date.month
    day = date.day
    
    if day <= 15:
        semimonthly = f"{year}-{month:02d}-SM1"
    else:
        semimonthly = f"{year}-{month:02d}-SM2"
    
    return semimonthly

# 2. 특정 날짜 이후의 다음 반월 기간 계산 함수
def get_next_semimonthly_period(date):
    """
    주어진 날짜 이후의 다음 반월 기간을 계산하는 함수
    """
    year = date.year
    month = date.month
    day = date.day
    
    if day <= 15:
        # 현재 상반월이면 같은 달의 하반월
        semimonthly = f"{year}-{month:02d}-SM2"
    else:
        # 현재 하반월이면 다음 달의 상반월
        if month == 12:
            # 12월 하반월이면 다음 해 1월 상반월
            semimonthly = f"{year+1}-01-SM1"
        else:
            semimonthly = f"{year}-{(month+1):02d}-SM1"
    
    return semimonthly

# 3. 반월 기간의 시작일과 종료일 계산 함수
def get_semimonthly_date_range(semimonthly_period):
    """
    반월 기간 문자열을 받아 시작일과 종료일을 계산하는 함수
    
    Parameters:
    -----------
    semimonthly_period : str
        "YYYY-MM-SM1" 또는 "YYYY-MM-SM2" 형식의 반월 기간
    
    Returns:
    --------
    tuple
        (시작일, 종료일) - datetime 객체
    """
    parts = semimonthly_period.split('-')
    year = int(parts[0])
    month = int(parts[1])
    half = parts[2]
    
    if half == "SM1":
        # 상반월 (1일~15일)
        start_date = pd.Timestamp(year=year, month=month, day=1)
        end_date = pd.Timestamp(year=year, month=month, day=15)
    else:
        # 하반월 (16일~말일)
        start_date = pd.Timestamp(year=year, month=month, day=16)
        _, last_day = calendar.monthrange(year, month)
        end_date = pd.Timestamp(year=year, month=month, day=last_day)
    
    return start_date, end_date

# 4. 다음 반월의 모든 날짜 목록 생성 함수
def get_next_semimonthly_dates(current_date, original_df):
    """
    현재 날짜 이후의 다음 반월 기간에 속하는 모든 영업일 목록을 반환하는 함수
    휴일(주말 및 공휴일)은 제외
    """
    # 다음 반월 기간 계산
    next_period = get_next_semimonthly_period(current_date)
    
    # 반월 기간의 시작일과 종료일 계산
    start_date, end_date = get_semimonthly_date_range(next_period)
    
    # 이 기간에 속하는 영업일(월~금, 휴일 제외) 선택
    business_days = []
    future_dates = original_df.index[original_df.index > current_date]
    
    for date in future_dates:
        if start_date <= date <= end_date and date.weekday() < 5 and not is_holiday(date):
            business_days.append(date)
    
    # 날짜가 없거나 부족하면 추가 로직 - 반월 기간 내에서만 합성 날짜 생성
    min_required_days = 5  # 최소 필요한 영업일 수
    if len(business_days) < min_required_days:
        logger.warning(f"Only {len(business_days)} business days found in the next semimonthly period. Creating synthetic dates.")
        
        # 마지막 실제 날짜 또는 시작일부터 시작
        if business_days:
            synthetic_date = business_days[-1] + pd.Timedelta(days=1)
        else:
            synthetic_date = max(current_date, start_date) + pd.Timedelta(days=1)
        
        # 반월 기간 내에서만 날짜 생성
        while len(business_days) < 15 and synthetic_date <= end_date:  # 반월 기간 초과 방지
            if synthetic_date.weekday() < 5 and not is_holiday(synthetic_date):  # 평일이고 휴일이 아닌 경우만 추가
                business_days.append(synthetic_date)
            synthetic_date += pd.Timedelta(days=1)
        
        logger.info(f"Created synthetic dates. Total business days: {len(business_days)} for period {next_period}")
    
    # 날짜순으로 정렬
    business_days.sort()
    
    return business_days, next_period

# 5. 다음 N 영업일 계산 함수
def get_next_n_business_days(current_date, original_df, n_days=23):
    """
    현재 날짜 이후의 n_days 영업일을 반환하는 함수 - 원본 데이터에 없는 미래 날짜도 생성
    휴일(주말 및 공휴일)은 제외
    """
    # 현재 날짜 이후의 데이터프레임에서 영업일 찾기
    future_df = original_df[original_df.index > current_date]
    
    # 필요한 수의 영업일 선택
    business_days = []
    
    # 먼저 데이터프레임에 있는 영업일 추가
    for date in future_df.index:
        if date.weekday() < 5 and not is_holiday(date):  # 월~금이고 휴일이 아닌 경우만 선택
            business_days.append(date)
        
        if len(business_days) >= n_days:
            break
    
    # 데이터프레임에서 충분한 날짜를 찾지 못한 경우 합성 날짜 생성
    if len(business_days) < n_days:
        # 마지막 날짜 또는 현재 날짜에서 시작
        last_date = business_days[-1] if business_days else current_date
        
        # 필요한 만큼 추가 날짜 생성
        current = last_date + pd.Timedelta(days=1)
        while len(business_days) < n_days:
            if current.weekday() < 5 and not is_holiday(current):  # 월~금이고 휴일이 아닌 경우만 포함
                business_days.append(current)
            current += pd.Timedelta(days=1)
    
    logger.info(f"Generated {len(business_days)} business days, excluding holidays")
    return business_days

# 6. 구간별 평균 가격 계산 및 점수 부여 함수
def calculate_interval_averages_and_scores(predictions, business_days, min_window_size=5):
    """
    다음 반월 기간에 대해 다양한 크기의 구간별 평균 가격을 계산하고 점수를 부여하는 함수
    - 반월 전체 영업일 수에 맞춰 윈도우 크기 범위 조정
    - global_rank 방식: 모든 구간을 비교해 전역적으로 가장 저렴한 구간에 점수 부여
    
    Parameters:
    -----------
    predictions : list
        날짜별 예측 가격 정보 (딕셔너리 리스트)
    business_days : list
        다음 반월의 영업일 목록
    min_window_size : int
        최소 고려할 윈도우 크기 (기본값: 3)
    
    Returns:
    -----------
    tuple
        (구간별 평균 가격 정보, 구간별 점수 정보, 분석 추가 정보)
    """
    import numpy as np
    
    # 예측 데이터를 날짜별로 정리
    predictions_dict = {pred['Date']: pred['Prediction'] for pred in predictions if pred['Date'] in business_days}
    
    # 날짜 순으로 정렬된 영업일 목록
    sorted_days = sorted(business_days)
    
    # 다음 반월 총 영업일 수 계산
    total_days = len(sorted_days)
    
    # 최소 윈도우 크기와 최대 윈도우 크기 설정 (최대는 반월 전체 일수)
    max_window_size = total_days
    
    # 고려할 모든 윈도우 크기 범위 생성
    window_sizes = range(min_window_size, max_window_size + 1)
    
    print(f"다음 반월 영업일: {total_days}일, 고려할 윈도우 크기: {list(window_sizes)}")
    
    # 각 윈도우 크기별 결과 저장
    interval_averages = {}
    
    # 모든 구간을 저장할 리스트
    all_intervals = []
    
    # 각 윈도우 크기에 대해 모든 가능한 구간 계산
    for window_size in window_sizes:
        window_results = []
        
        # 가능한 모든 시작점에 대해 윈도우 평균 계산
        for i in range(len(sorted_days) - window_size + 1):
            interval_days = sorted_days[i:i+window_size]
            
            # 모든 날짜에 예측 가격이 있는지 확인
            if all(day in predictions_dict for day in interval_days):
                avg_price = np.mean([predictions_dict[day] for day in interval_days])
                
                interval_info = {
                    'start_date': interval_days[0],
                    'end_date': interval_days[-1],
                    'days': window_size,
                    'avg_price': avg_price,
                    'dates': interval_days.copy()
                }
                
                window_results.append(interval_info)
                all_intervals.append(interval_info)  # 모든 구간 목록에도 추가
        
        # 해당 윈도우 크기에 대한 결과 저장 (참고용)
        if window_results:
            # 평균 가격 기준으로 정렬
            window_results.sort(key=lambda x: x['avg_price'])
            interval_averages[window_size] = window_results
    
    # 구간 점수 계산을 위한 딕셔너리
    interval_scores = {}
    
    # Global Rank 전략: 모든 구간을 통합하여 가격 기준으로 정렬
    all_intervals.sort(key=lambda x: x['avg_price'])
    
    # 상위 3개 구간에만 점수 부여 (전체 중에서)
    for i, interval in enumerate(all_intervals[:min(3, len(all_intervals))]):
        score = 3 - i  # 1등: 3점, 2등: 2점, 3등: 1점
        
        # 구간 식별을 위한 키 생성 (문자열 키로 변경)
        interval_key = f"{format_date(interval['start_date'])}-{format_date(interval['end_date'])}"
        
        # 점수 정보 저장
        interval_scores[interval_key] = {
            'start_date': format_date(interval['start_date']),  # 형식 적용
            'end_date': format_date(interval['end_date']),      # 형식 적용
            'days': interval['days'],
            'avg_price': interval['avg_price'],
            'dates': [format_date(d) for d in interval['dates']],  # 날짜 목록도 형식 적용
            'score': score,
            'rank': i + 1
        }
    
    # 분석 정보 추가
    analysis_info = {
        'total_days': total_days,
        'window_sizes': list(window_sizes),
        'total_intervals': len(all_intervals),
        'min_avg_price': min([interval['avg_price'] for interval in all_intervals]) if all_intervals else None,
        'max_avg_price': max([interval['avg_price'] for interval in all_intervals]) if all_intervals else None
    }
    
    # 결과 출력 (참고용)
    if interval_scores:
        top_interval = max(interval_scores.values(), key=lambda x: x['score'])
        print(f"\n최고 점수 구간: {top_interval['days']}일 구간 ({format_date(top_interval['start_date'])} ~ {format_date(top_interval['end_date'])})")
        print(f"점수: {top_interval['score']}, 순위: {top_interval['rank']}, 평균가: {top_interval['avg_price']:.2f}")
    
    return interval_averages, interval_scores, analysis_info

# 7. 두 구매 방법의 결과 비교 함수
def decide_purchase_interval(interval_scores):
    """
    점수가 부여된 구간들 중에서 최종 구매 구간을 결정하는 함수
    - 점수가 가장 높은 구간 선택
    - 동점인 경우 평균 가격이 더 낮은 구간 선택
    
    Parameters:
    -----------
    interval_scores : dict
        구간별 점수 정보
    
    Returns:
    -----------
    dict
        최종 선택된 구매 구간 정보
    """
    if not interval_scores:
        return None
    
    # 점수가 가장 높은 구간 선택
    max_score = max(interval['score'] for interval in interval_scores.values())
    
    # 최고 점수를 가진 모든 구간 찾기
    top_intervals = [interval for interval in interval_scores.values() 
                    if interval['score'] == max_score]
    
    # 동점이 있는 경우, 평균 가격이 더 낮은 구간 선택
    if len(top_intervals) > 1:
        best_interval = min(top_intervals, key=lambda x: x['avg_price'])
        best_interval['selection_reason'] = "최고 점수 중 최저 평균가 구간"
    else:
        best_interval = top_intervals[0]
        best_interval['selection_reason'] = "최고 점수 구간"
    
    return best_interval

#######################################################################
# 특성 선택 함수
#######################################################################

def calculate_group_vif(df, variables):
    """그룹 내 변수들의 VIF 계산"""
    # 변수가 한 개 이하면 VIF 계산 불가
    if len(variables) <= 1:
        return pd.DataFrame({
            "Feature": variables,
            "VIF": [1.0] * len(variables)
        })
    
    # 모든 변수가 데이터프레임에 존재하는지 확인
    available_vars = [var for var in variables if var in df.columns]
    if len(available_vars) <= 1:
        return pd.DataFrame({
            "Feature": available_vars,
            "VIF": [1.0] * len(available_vars)
        })
    
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = available_vars
        vif_data["VIF"] = [variance_inflation_factor(df[available_vars].values, i) 
                          for i in range(len(available_vars))]
        return vif_data.sort_values('VIF', ascending=False)
    except Exception as e:
        logger.error(f"Error calculating VIF: {str(e)}")
        # 오류 발생 시 기본값 반환
        return pd.DataFrame({
            "Feature": available_vars,
            "VIF": [float('nan')] * len(available_vars)
        })

def analyze_group_correlations(df, variable_groups, target_col='MOPJ'):
    """그룹별 상관관계 분석"""
    logger.info("Analyzing correlations for each group:")
    group_correlations = {}
    
    for group_name, variables in variable_groups.items():
        # 각 그룹의 변수들과 타겟 변수의 상관관계 계산
        # 해당 그룹의 변수들이 데이터프레임에 존재하는지 확인
        available_vars = [var for var in variables if var in df.columns]
        if not available_vars:
            logger.warning(f"Warning: No variables from {group_name} group found in dataframe")
            continue
            
        if target_col not in df.columns:
            logger.warning(f"Warning: Target column {target_col} not found in dataframe")
            continue
            
        correlations = df[available_vars].corrwith(df[target_col]).abs().sort_values(ascending=False)
        group_correlations[group_name] = correlations
        
        logger.info(f"\n{group_name} group correlations with {target_col}:")
        logger.info(str(correlations))
    
    return group_correlations

def select_features_from_groups(df, variable_groups, target_col='MOPJ', vif_threshold=50.0, corr_threshold=0.8):
    """각 그룹에서 대표 변수 선택"""
    selected_features = []
    selection_process = {}
    
    logger.info(f"\nCorrelation threshold: {corr_threshold}")
    
    for group_name, variables in variable_groups.items():
        logger.info(f"\nProcessing {group_name} group:")
        
        # 해당 그룹의 변수들이 df에 존재하는지 확인
        available_vars = [var for var in variables if var in df.columns]
        if not available_vars:
            logger.warning(f"Warning: No variables from {group_name} group found in dataframe")
            continue
            
        if target_col not in df.columns:
            logger.warning(f"Warning: Target column {target_col} not found in dataframe")
            continue
        
        # 그룹 내 상관관계 계산
        correlations = df[available_vars].corrwith(df[target_col]).abs().sort_values(ascending=False)
        logger.info(f"\nCorrelations with {target_col}:")
        logger.info(str(correlations))
        
        # 상관관계가 임계값 이상인 변수만 필터링
        high_corr_vars = correlations[correlations >= corr_threshold].index.tolist()
        
        if not high_corr_vars:
            logger.warning(f"Warning: No variables in {group_name} group meet the correlation threshold of {corr_threshold}")
            continue
        
        # 상관관계 임계값을 만족하는 변수들에 대해 VIF 계산
        if len(high_corr_vars) > 1:
            vif_data = calculate_group_vif(df[high_corr_vars], high_corr_vars)
            logger.info(f"\nVIF values for {group_name} group (high correlation vars only):")
            logger.info(str(vif_data))
            
            # VIF 기준 적용하여 다중공선성 낮은 변수 선택
            low_vif_vars = vif_data[vif_data['VIF'] < vif_threshold]['Feature'].tolist()
            
            if low_vif_vars:
                # 낮은 VIF 변수들 중 상관관계가 가장 높은 변수 선택
                for var in correlations.index:
                    if var in low_vif_vars:
                        selected_var = var
                        break
                else:
                    selected_var = high_corr_vars[0]
            else:
                selected_var = high_corr_vars[0]
        else:
            selected_var = high_corr_vars[0]
            vif_data = pd.DataFrame({"Feature": [selected_var], "VIF": [1.0]})
        
        # 선택된 변수가 상관관계 임계값을 만족하는지 확인 (안전장치)
        if correlations[selected_var] >= corr_threshold:
            selected_features.append(selected_var)
            
            selection_process[group_name] = {
                'selected_variable': selected_var,
                'correlation': correlations[selected_var],
                'all_correlations': correlations.to_dict(),
                'vif_data': vif_data.to_dict() if not vif_data.empty else {},
                'high_corr_vars': high_corr_vars
            }
            
            logger.info(f"\nSelected variable from {group_name}: {selected_var} (corr: {correlations[selected_var]:.4f})")
        else:
            logger.info(f"\nNo variable selected from {group_name}: correlation threshold not met")
    
    # 상관관계 기준 재확인 (최종 안전장치)
    final_features = []
    for feature in selected_features:
        corr = abs(df[feature].corr(df[target_col]))
        if corr >= corr_threshold:
            final_features.append(feature)
            logger.info(f"Final selection: {feature} (corr: {corr:.4f})")
        else:
            logger.info(f"Excluded: {feature} (corr: {corr:.4f}) - below threshold")
    
    # 타겟 컬럼이 포함되어 있지 않으면 추가
    if target_col not in final_features:
        final_features.append(target_col)
        logger.info(f"Added target column: {target_col}")
    
    # 최소 특성 수 확인
    if len(final_features) < 3:
        logger.warning(f"Selected features ({len(final_features)}) < 3, lowering threshold to 0.5")
        return select_features_from_groups(df, variable_groups, target_col, vif_threshold, 0.5)
    
    return final_features, selection_process

def optimize_hyperparameters_semimonthly_kfold(train_data, input_size, target_col_idx, device, current_period, n_trials=30, k_folds=5, use_cache=True):
    """
    시계열 K-fold 교차 검증을 사용하여 반월별 데이터에 대한 하이퍼파라미터 최적화
    """
    logger.info(f"\n===== {current_period} 하이퍼파라미터 최적화 시작 (시계열 {k_folds}-fold 교차 검증) =====")
    
    # 캐시 파일 경로
    cache_file = f"models/hyperparams_kfold_{current_period.replace('-', '_')}.json"
    
    # 캐시 확인
    if use_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_params = json.load(f)
            logger.info(f"{current_period} 캐시된 하이퍼파라미터 로드 완료")
            return cached_params
        except Exception as e:
            logger.error(f"캐시 파일 로드 오류: {str(e)}")
    
    # 기본 하이퍼파라미터 정의 (최적화 실패 시 사용)
    default_params = {
        'sequence_length': 20,
        'hidden_size': 193,
        'num_layers': 6,
        'dropout': 0.2548844201860255,
        'batch_size': 77,
        'learning_rate': 0.00012082451343807303,
        'num_epochs': 107,
        'loss_alpha': 0.506378255841371,  # MSE weight
        'loss_beta': 0.17775383895727725,  # Volatility weight
        'loss_gamma': 0.07133778859895412, # Directional weight
        'loss_delta': 0.07027938312247926, # Continuity weight
        'patience': 26,
        'warmup_steps': 382,
        'lr_factor': 0.49185859987249164,
        'lr_patience': 8,
        'min_lr': 1.1304817036887926e-07
    }
    
    # 데이터 길이 확인 - 충분하지 않으면 바로 기본값 반환
    MIN_DATA_SIZE = 100
    if len(train_data) < MIN_DATA_SIZE:
        logger.warning(f"훈련 데이터가 너무 적습니다 ({len(train_data)} 데이터 포인트 < {MIN_DATA_SIZE}). 기본 파라미터를 사용합니다.")
        return default_params
    
    # K-fold 분할 로직
    predict_window = 23  # 예측 윈도우 크기
    min_fold_size = 20 + predict_window + 5  # 최소 시퀀스 길이 + 예측 윈도우 + 여유
    max_possible_folds = len(train_data) // min_fold_size
    
    if max_possible_folds < 2:
        logger.warning(f"데이터가 충분하지 않아 k-fold를 수행할 수 없습니다 (가능한 fold: {max_possible_folds} < 2). 기본 파라미터를 사용합니다.")
        return default_params
    
    # 실제 사용 가능한 fold 수 조정
    k_folds = min(k_folds, max_possible_folds)
    fold_size = len(train_data) // (k_folds + 1)  # +1은 예측 윈도우를 위한 추가 부분

    logger.info(f"데이터 크기: {len(train_data)}, Fold 수: {k_folds}, 각 Fold 크기: {fold_size}")

    # fold 분할을 위한 인덱스 생성
    folds = []
    for i in range(k_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        
        train_indices = list(range(0, test_start)) + list(range(test_end, len(train_data)))
        test_indices = list(range(test_start, test_end))
        
        folds.append((train_indices, test_indices))
    
    # Optuna 목적 함수 정의
    def objective(trial):
        # 하이퍼파라미터 범위 수정 - 시퀀스 길이 최대값 제한
        max_seq_length = min(fold_size - predict_window - 5, 60)
        
        # 최소 시퀀스 길이도 제한
        min_seq_length = min(10, max_seq_length)
        
        if max_seq_length <= min_seq_length:
            logger.warning(f"시퀀스 길이 범위가 너무 제한적입니다 (min={min_seq_length}, max={max_seq_length}). 해당 trial 건너뛰기.")
            return float('inf')
        
        params = {
            'sequence_length': trial.suggest_int('sequence_length', min_seq_length, max_seq_length),
            'hidden_size': trial.suggest_int('hidden_size', 32, 256),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'batch_size': trial.suggest_int('batch_size', 16, min(128, len(train_data))),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'num_epochs': trial.suggest_int('num_epochs', 50, 200),
            'patience': trial.suggest_int('patience', 10, 30),
            'warmup_steps': trial.suggest_int('warmup_steps', 100, 1000),
            'lr_factor': trial.suggest_float('lr_factor', 0.1, 0.5),
            'lr_patience': trial.suggest_int('lr_patience', 3, 10),
            'min_lr': trial.suggest_float('min_lr', 1e-7, 1e-5, log=True),
            'loss_alpha': trial.suggest_float('loss_alpha', 0.5, 0.9),
            'loss_beta': trial.suggest_float('loss_beta', 0.1, 0.3),
            'loss_gamma': trial.suggest_float('loss_gamma', 0.05, 0.2),
            'loss_delta': trial.suggest_float('loss_delta', 0.01, 0.1)
        }
        
        # K-fold 교차 검증
        fold_losses = []
        valid_fold_count = 0
        
        for fold_idx, (train_indices, test_indices) in enumerate(folds):
            try:
                # 시퀀스 길이가 fold 크기보다 크면 건너뛰기
                if params['sequence_length'] >= len(test_indices):
                    logger.warning(f"Fold {fold_idx+1}: 시퀀스 길이({params['sequence_length']})가 테스트 데이터({len(test_indices)})보다 큽니다.")
                    continue
                
                # fold별 훈련/테스트 데이터 준비
                fold_train_data = train_data[train_indices]
                fold_test_data = train_data[test_indices]
                
                # 데이터 준비
                X_train, y_train, prev_train, X_val, y_val, prev_val = prepare_data(
                    fold_train_data, fold_test_data, params['sequence_length'],
                    predict_window, target_col_idx, augment=False
                )
                
                # 데이터가 충분한지 확인
                if len(X_train) < params['batch_size'] or len(X_val) < 1:
                    logger.warning(f"Fold {fold_idx+1}: 데이터 불충분 (훈련: {len(X_train)}, 검증: {len(X_val)})")
                    continue
                
                # 데이터셋 및 로더 생성
                train_dataset = TimeSeriesDataset(X_train, y_train, device, prev_train)
                batch_size = min(params['batch_size'], len(X_train))
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    worker_init_fn=seed_worker,
                    generator=g
                )
                
                val_dataset = TimeSeriesDataset(X_val, y_val, device, prev_val)
                val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
                
                # 모델 생성
                model = ImprovedLSTMPredictor(
                    input_size=input_size,
                    hidden_size=params['hidden_size'],
                    num_layers=params['num_layers'],
                    dropout=params['dropout'],
                    output_size=predict_window
                ).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

                # best_val_loss 변수 명시적 정의
                best_val_loss = float('inf')
                patience_counter = 0
                
                for epoch in range(params['num_epochs']):
                    # 학습
                    model.train()
                    train_loss = 0
                    for X_batch, y_batch, prev_batch in train_loader:
                        optimizer.zero_grad()
                        y_pred = model(X_batch, prev_batch)
                        loss, _ = composite_loss(y_pred, y_batch, prev_batch,
                                            alpha=params['loss_alpha'],
                                            beta=params['loss_beta'],
                                            gamma=params['loss_gamma'],
                                            delta=params['loss_delta'])
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        train_loss += loss.item()
                    
                    # 검증
                    model.eval()
                    val_loss = 0
                    
                    with torch.no_grad():
                        for X_batch, y_batch, prev_batch in val_loader:
                            y_pred = model(X_batch, prev_batch)
                            loss, _ = composite_loss(y_pred, y_batch, prev_batch,
                                                alpha=params['loss_alpha'],
                                                beta=params['loss_beta'],
                                                gamma=params['loss_gamma'],
                                                delta=params['loss_delta'])
                            val_loss += loss.item()
                        
                        val_loss /= len(val_loader)
                        
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                        
                        if patience_counter >= params['patience']:
                            break
                
                valid_fold_count += 1
                fold_losses.append(best_val_loss)
                
            except Exception as e:
                logger.error(f"Error in fold {fold_idx+1}: {str(e)}")
                continue
        
        # 모든 fold가 실패한 경우 매우 큰 손실값 반환
        if not fold_losses:
            logger.warning("모든 fold가 실패했습니다. 이 파라미터 조합은 건너뜁니다.")
            return float('inf')
        
        # 성공한 fold의 평균 손실값 반환
        return sum(fold_losses) / len(fold_losses)
    
    # Optuna 최적화 시도
    try:
        import optuna
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        # 최적 하이퍼파라미터
        if study.best_trial.value == float('inf'):
            logger.warning(f"모든 trial이 실패했습니다. 기본 하이퍼파라미터를 사용합니다.")
            return default_params
            
        best_params = study.best_params
        logger.info(f"\n{current_period} 최적 하이퍼파라미터 (K-fold):")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        # 모든 필수 키가 있는지 확인
        required_keys = ['sequence_length', 'hidden_size', 'num_layers', 'dropout', 
                        'batch_size', 'learning_rate', 'num_epochs', 'patience',
                        'warmup_steps', 'lr_factor', 'lr_patience', 'min_lr',
                        'loss_alpha', 'loss_beta', 'loss_gamma', 'loss_delta']
        
        for key in required_keys:
            if key not in best_params:
                # 누락된 키가 있으면 기본값 할당
                if key == 'warmup_steps':
                    best_params[key] = 382
                elif key == 'lr_factor':
                    best_params[key] = 0.49
                elif key == 'lr_patience':
                    best_params[key] = 8
                elif key == 'min_lr':
                    best_params[key] = 1e-7
                elif key == 'loss_gamma':
                    best_params[key] = 0.07
                elif key == 'loss_delta':
                    best_params[key] = 0.07
                else:
                    best_params[key] = default_params[key]
        
        # 캐시에 저장
        with open(cache_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        logger.info(f"하이퍼파라미터가 {cache_file}에 저장되었습니다.")
        
        return best_params
        
    except Exception as e:
        logger.error(f"하이퍼파라미터 최적화 오류: {str(e)}")
        traceback.print_exc()
        
        # 오류 시 기본 하이퍼파라미터 반환
        return default_params

#######################################################################
# 시각화 함수
#######################################################################

def get_global_y_range(original_df, test_dates, predict_window):
    """
    테스트 구간의 모든 MOPJ 값을 기반으로 전역 y축 범위를 계산합니다.
    
    Args:
        original_df: 원본 데이터프레임
        test_dates: 테스트 날짜 배열
        predict_window: 예측 기간
    
    Returns:
        tuple: (y_min, y_max) 전역 범위 값
    """
    # 테스트 구간 데이터 추출
    test_values = []
    
    # 테스트 데이터의 실제 값 수집
    for date in test_dates:
        if date in original_df.index and not pd.isna(original_df.loc[date, 'MOPJ']):
            test_values.append(original_df.loc[date, 'MOPJ'])
    
    # 안전장치: 데이터가 없으면 None 반환
    if not test_values:
        return None, None
    
    # 최소/최대 계산 (약간의 마진 추가)
    y_min = min(test_values) * 0.95
    y_max = max(test_values) * 1.05
    
    return y_min, y_max

def visualize_attention_weights(model, features, prev_value, sequence_start_date, feature_names=None):
    """모델의 어텐션 가중치를 시각화하는 함수"""
    model.eval()
    
    # 특성 이름이 없으면 인덱스로 생성
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(features.shape[2])]
    else:
        # 특성 수에 맞게 조정
        feature_names = feature_names[:features.shape[2]]
    
    # 텐서가 아니면 변환
    if not isinstance(features, torch.Tensor):
        features = torch.FloatTensor(features).to(next(model.parameters()).device)
    
    # prev_value 처리
    if prev_value is not None:
        if not isinstance(prev_value, torch.Tensor):
            try:
                prev_value = float(prev_value)
                prev_value = torch.FloatTensor([prev_value]).to(next(model.parameters()).device)
            except (TypeError, ValueError):
                logger.warning("Warning: prev_value를 숫자로 변환할 수 없습니다. 0으로 대체합니다.")
                prev_value = torch.FloatTensor([0.0]).to(next(model.parameters()).device)
    
    # 시퀀스 길이
    seq_len = features.shape[1]
    
    # 날짜 라벨 생성 (시퀀스 시작일로부터)
    date_labels = []
    for i in range(seq_len):
        try:
            date = sequence_start_date - timedelta(days=seq_len-i-1)
            date_labels.append(format_date(date, '%Y-%m-%d'))
        except:
            date_labels.append(f"T-{seq_len-i-1}")
    
    # 1x2 그래프 생성 (특성 중요도, 시간 중요도)
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle(f"Model Importance Analysis - {format_date(sequence_start_date, '%Y-%m-%d')}", 
                fontsize=16)
    
    # 특성 중요도 계산
    feature_importance = np.mean(np.abs(features[0].cpu().numpy()), axis=0)
    
    # 정규화
    if np.sum(feature_importance) > 0:
        feature_importance = feature_importance / np.sum(feature_importance)
    
    # 특성 중요도를 내림차순으로 정렬
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = feature_importance[sorted_idx]
    
    # 상위 10개 특성만 표시
    top_n = min(10, len(sorted_features))
    
    # 플롯 1: 특성별 중요도 (수평 막대 그래프)
    ax1 = axes[0]
    
    try:
        # 수평 막대 그래프로 표시
        y_pos = range(top_n)
        ax1.barh(y_pos, sorted_importance[:top_n], color='#3498db')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sorted_features[:top_n])
        ax1.set_title("Top Feature Importance")
        ax1.set_xlabel("Relative Importance")
        
        # 중요도 값 표시
        for i, v in enumerate(sorted_importance[:top_n]):
            ax1.text(v + 0.01, i, f"{v:.3f}", va='center')
    except Exception as e:
        logger.error(f"Feature importance visualization error: {str(e)}")
        ax1.text(0.5, 0.5, "Visualization error", ha='center', va='center')
    
    # 플롯 2: 시간적 중요도
    ax2 = axes[1]
    
    # 각 시점의 평균 절대값으로 시간적 중요도 추정
    temporal_importance = np.mean(np.abs(features[0].cpu().numpy()), axis=1)
    if np.sum(temporal_importance) > 0:
        temporal_importance = temporal_importance / np.sum(temporal_importance)
    
    try:
        # 시간적 중요도 표시 - 막대 그래프
        ax2.bar(range(len(date_labels)), temporal_importance, color='#2ecc71')
        ax2.set_xticks(range(len(date_labels)))
        ax2.set_xticklabels(date_labels, rotation=45, ha='right')
        ax2.set_title("Time Sequence Importance")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Relative Importance")
        
        # 마지막 시점 강조
        ax2.bar(len(date_labels)-1, temporal_importance[-1], color='#e74c3c')
    except Exception as e:
        logger.error(f"Time importance visualization error: {str(e)}")
        ax2.text(0.5, 0.5, "Visualization error", ha='center', va='center')
    
    plt.tight_layout()
    
    # 이미지를 메모리에 저장
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=600)
    plt.close()
    img_buf.seek(0)
    
    # Base64로 인코딩
    img_str = base64.b64encode(img_buf.read()).decode('utf-8')
    
    # 파일 저장
    attn_dir = ATTENTION_DIR
    os.makedirs(attn_dir, exist_ok=True)
    
    try:
        filename = os.path.join(attn_dir, f"attention_{format_date(sequence_start_date, '%Y%m%d')}.png")
        with open(filename, 'wb') as f:
            f.write(base64.b64decode(img_str))
    except Exception as e:
        logger.error(f"Error saving attention image: {str(e)}")
        filename = None
    
    return filename, img_str, {
        'feature_importance': dict(zip(sorted_features, sorted_importance.tolist())),
        'temporal_importance': dict(zip(date_labels, temporal_importance.tolist()))
    }

def plot_prediction_basic(sequence_df, sequence_start_date, start_day_value, 
                         f1, accuracy, mape, weighted_score_pct, 
                         save_prefix=PLOT_DIR, title_prefix="Basic Prediction Graph",
                         y_min=None, y_max=None):
    """기본 예측 그래프 시각화"""
    try:
        logger.info(f"Creating basic prediction graph for {format_date(sequence_start_date)}")
        
        # 유효한 실제값 / 예측값
        valid_df = sequence_df.dropna(subset=['Actual'])
        pred_df = sequence_df.dropna(subset=['Prediction'])
        
        # DataFrame의 날짜 열이 문자열인 경우 날짜 객체로 변환
        if 'Date' in sequence_df.columns and isinstance(sequence_df['Date'].iloc[0], str):
            sequence_df['Date'] = pd.to_datetime(sequence_df['Date'])
            valid_df['Date'] = pd.to_datetime(valid_df['Date'])
            pred_df['Date'] = pd.to_datetime(pred_df['Date'])
        
        # 그래프 생성
        fig = plt.figure(figsize=(16, 9))
        plt.subplots_adjust(top=0.85)
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3, figure=fig)
        
        # 그래프 타이틀과 서브타이틀
        if isinstance(sequence_start_date, str):
            main_title = f"{title_prefix} - {sequence_start_date}"
        else:
            main_title = f"{title_prefix} - {sequence_start_date.strftime('%Y-%m-%d')}"
        subtitle = f"F1: {f1:.2f}, Acc: {accuracy:.2f}%, MAPE: {mape:.2f}%, Score: {weighted_score_pct:.2f}%"

        # 별도 텍스트로 제목 추가
        fig.text(0.5, 0.94, main_title, ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.90, subtitle, ha='center', fontsize=12)
        
        # (1) 상단: Real vs Pred
        ax1 = fig.add_subplot(gs[0])
        ax1.set_title("Prediction vs Actual")
        ax1.grid(True, linestyle='--', alpha=0.5)

        if y_min is not None and y_max is not None:
            ax1.set_ylim(y_min, y_max)
        
        # sequence_start_date가 문자열인 경우 datetime으로 변환
        if isinstance(sequence_start_date, str):
            sequence_start_date = pd.to_datetime(sequence_start_date)
        
        # 날짜별로 x축을 구성할 때 쓰일 시계열
        real_dates = [sequence_start_date] + valid_df['Date'].tolist()
        real_values = [start_day_value] + valid_df['Actual'].tolist()
        
        pred_dates = [sequence_start_date] + pred_df['Date'].tolist()
        pred_values = [start_day_value] + pred_df['Prediction'].tolist()
        
        ax1.plot(real_dates, real_values,
                marker='o', color='blue', label='Actual')
        ax1.plot(pred_dates, pred_values,
                marker='o', color='red', label='Predicted')
        
        ax1.set_xlabel("")
        ax1.set_ylabel("Price")
        ax1.legend()
        
        # 배경 색칠 (방향성 일치 여부)
        real_index_set = set(valid_df['Date'])
        pred_index_set = set(pred_df['Date'])
        common_dates = sorted(list(real_index_set.intersection(pred_index_set)))
        
        if len(common_dates) > 1:
            if sequence_start_date not in common_dates:
                common_dates.insert(0, sequence_start_date)

            for i in range(len(common_dates) - 1):
                curr_date = common_dates[i]
                next_date = common_dates[i+1]

                # 실제값: curr_date, next_date
                if curr_date == sequence_start_date:
                    actual_curr = start_day_value
                else:
                    actual_curr = valid_df.loc[valid_df['Date'] == curr_date, 'Actual'].values[0]

                actual_next = valid_df.loc[valid_df['Date'] == next_date, 'Actual']
                if not actual_next.empty:
                    actual_next = actual_next.values[0]
                else:
                    continue

                # 예측값: curr_date, next_date
                if curr_date == sequence_start_date:
                    pred_curr = start_day_value
                else:
                    pred_curr = pred_df.loc[pred_df['Date'] == curr_date, 'Prediction'].values[0]

                pred_next = pred_df.loc[pred_df['Date'] == next_date, 'Prediction']
                if not pred_next.empty:
                    pred_next = pred_next.values[0]
                else:
                    continue

                # 방향 계산
                actual_dir = np.sign(actual_next - actual_curr)
                pred_dir = np.sign(pred_next - pred_curr)

                color = 'blue' if actual_dir == pred_dir else 'red'
                ax1.axvspan(curr_date, next_date, color=color, alpha=0.1)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # (2) 하단: 절대 오차(Bar)
        ax2 = fig.add_subplot(gs[1])
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        error_dates = []
        error_values = []
        for d in pred_df['Date']:
            if d in valid_df['Date'].values:
                a_val = valid_df.loc[valid_df['Date'] == d, 'Actual'].values[0]
                p_val = pred_df.loc[pred_df['Date'] == d, 'Prediction'].values[0]
                error_dates.append(d)
                error_values.append(abs(a_val - p_val))
        
        if error_dates and error_values:  # 값이 있는지 확인
            ax2.bar(error_dates, error_values, width=0.6, color='salmon', alpha=0.7)
        
        ax2.set_title("Absolute Error (|Real - Pred|)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Error")
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 저장 전 디렉토리 확인 및 생성
        logger.info(f"Saving graph to directory: {save_prefix}")
        os.makedirs(save_prefix, exist_ok=True)
        
        # 파일 경로 생성
        if isinstance(sequence_start_date, str):
            date_str = pd.to_datetime(sequence_start_date).strftime('%Y%m%d')
        else:
            date_str = sequence_start_date.strftime('%Y%m%d')
        
        filename = f"basic_prediction_{date_str}.png"
        full_path = os.path.join(save_prefix, filename)
        
        # 이미지를 메모리에 저장
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300)
        img_buf.seek(0)
        
        # Base64로 인코딩
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        
        # 저장
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Basic prediction graph saved: {full_path}")
        
        return full_path, img_str
        
    except Exception as e:
        logger.error(f"Error in graph creation: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None
    
def plot_moving_average_analysis(ma_results, sequence_start_date, save_prefix=MA_PLOT_DIR,
                                title_prefix="Moving Average Analysis", y_min=None, y_max=None):
    """이동평균 분석 시각화"""
    try:
        # ma_results 형식: {'ma5': [{'date': '...', 'prediction': X, 'actual': Y, 'ma': Z}, ...], 'ma10': [...]}
        windows = sorted(ma_results.keys())
        
        fig = plt.figure(figsize=(12, 4 * len(windows)))
        
        if isinstance(sequence_start_date, str):
            title = f"{title_prefix} Starting {sequence_start_date}"
        else:
            title = f"{title_prefix} Starting {sequence_start_date.strftime('%Y-%m-%d')}"
            
        fig.suptitle(title, fontsize=16)
        
        for idx, window_key in enumerate(windows):
            window_num = window_key.replace('ma', '')
            ax = fig.add_subplot(len(windows), 1, idx+1)
            
            window_data = ma_results[window_key]
            
            # 날짜, 예측, 실제값, MA 추출
            dates = []
            predictions = []
            actuals = []
            ma_preds = []
            
            for item in window_data:
                if isinstance(item['date'], str):
                    dates.append(pd.to_datetime(item['date']))
                else:
                    dates.append(item['date'])
                    
                predictions.append(item['prediction'])
                actuals.append(item['actual'])
                ma_preds.append(item['ma'])
            
            # y축 범위 설정
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
            
            # 원본 실제값 vs 예측값 (옅게)
            ax.plot(dates, actuals, marker='o', color='blue', alpha=0.3, label='Actual')
            ax.plot(dates, predictions, marker='o', color='red', alpha=0.3, label='Predicted')
            
            # 이동평균
            # 실제값(actuals)과 이동평균(ma_preds) 모두 None이 아닌 인덱스를 선택
            valid_indices = [
                i for i in range(len(ma_preds))
                if (ma_preds[i] is not None and actuals[i] is not None)
            ]

            if valid_indices:
                valid_dates = [dates[i] for i in valid_indices]
                valid_ma = [ma_preds[i] for i in valid_indices]
                valid_actuals = [actuals[i] for i in valid_indices]
                
                # 배열로 변환
                valid_actuals_arr = np.array(valid_actuals)
                valid_ma_arr = np.array(valid_ma)
                
                # 실제값이 0인 항목은 제외하여 MAPE 계산
                non_zero_mask = valid_actuals_arr != 0
                if np.sum(non_zero_mask) > 0:
                    ma_mape = np.mean(np.abs((valid_actuals_arr[non_zero_mask] - valid_ma_arr[non_zero_mask]) /
                                            valid_actuals_arr[non_zero_mask])) * 100
                else:
                    ma_mape = 0.0
                
                ax.set_title(f"MA-{window_num} Analysis (MAPE: {ma_mape:.2f}%, Count: {len(valid_indices)})")
            else:
                ax.set_title(f"MA-{window_num} Analysis (Insufficient data)")
            
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
        
        plt.tight_layout()
        
        # 저장
        os.makedirs(save_prefix, exist_ok=True)
        
        if isinstance(sequence_start_date, str):
            date_str = pd.to_datetime(sequence_start_date).strftime('%Y%m%d')
        else:
            date_str = sequence_start_date.strftime('%Y%m%d')
            
        filename = os.path.join(save_prefix, f"ma_analysis_{date_str}.png")
        
        # 이미지를 메모리에 저장
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300)
        img_buf.seek(0)
        
        # Base64로 인코딩
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        
        # 파일 저장
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Moving Average graph saved: {filename}")
        return filename, img_str
        
    except Exception as e:
        logger.error(f"Error in moving average visualization: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def compute_performance_metrics(sequence_df, start_day_value):
    """
    주어진 시퀀스 DataFrame과 시작일 실제값을 이용해 성능 지표를 계산하는 함수.
    sequence_df에는 'Date', 'Prediction', 'Actual' 컬럼이 포함되어 있어야 함.
    """
    try:
        # 시퀀스 DataFrame 처리
        if 'Date' in sequence_df.columns and isinstance(sequence_df['Date'].iloc[0], str):
            sequence_df['Date'] = pd.to_datetime(sequence_df['Date'])
            
        # 실제값이 있는 행만 사용하고 날짜순으로 정렬
        valid_df = sequence_df.dropna(subset=['Actual']).sort_values('Date')
        
        if len(valid_df) < 1:
            return {
                'f1': 0.0,
                'accuracy': 0.0,
                'mape': 0.0,
                'weighted_score': 0.0,
                'cosine_similarity': None,
                'f1_report': "Insufficient data"
            }

        # 시작일의 실제값을 포함하여 예측 및 실제 배열 생성
        actual_vals = np.concatenate([[start_day_value], valid_df['Actual'].values])
        pred_vals = np.concatenate([[start_day_value], valid_df['Prediction'].values])
        
        # F1 점수 계산 (방향성)
        f1, f1_report = calculate_f1_score(actual_vals, pred_vals)
        
        # 방향성 정확도
        direction_accuracy = calculate_direction_accuracy(actual_vals, pred_vals)
        
        # 가중치 점수 계산 (시작값은 제외)
        weighted_score, max_score = calculate_direction_weighted_score(actual_vals[1:], pred_vals[1:])
        weighted_score_pct = (weighted_score / max_score) * 100 if max_score > 0 else 0.0
        
        # MAPE 계산 (시작값 제외)
        mape = calculate_mape(actual_vals[1:], pred_vals[1:])
        
        # 코사인 유사도 계산
        if len(actual_vals) > 1 and len(pred_vals) > 1:
            diff_actual = np.diff(actual_vals)
            diff_pred = np.diff(pred_vals)
            norm_actual = np.linalg.norm(diff_actual)
            norm_pred = np.linalg.norm(diff_pred)
            if norm_actual > 0 and norm_pred > 0:
                cosine_similarity = np.dot(diff_actual, diff_pred) / (norm_actual * norm_pred)
            else:
                cosine_similarity = None
        else:
            cosine_similarity = None
            
        return {
            'f1': f1,
            'accuracy': direction_accuracy,
            'mape': mape,
            'weighted_score': weighted_score_pct,
            'cosine_similarity': cosine_similarity,
            'f1_report': f1_report
        }
        
    except Exception as e:
        logger.error(f"Error computing performance metrics: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'f1': 0.0,
            'accuracy': 0.0,
            'mape': 0.0,
            'weighted_score': 0.0,
            'cosine_similarity': None,
            'f1_report': f"Error: {str(e)}"
        }

def calculate_f1_score(actual, predicted):
    """방향성 예측의 F1 점수 계산"""
    actual_directions = np.sign(np.diff(actual))
    predicted_directions = np.sign(np.diff(predicted))

    if len(actual_directions) < 2:
        return 0.0, "Insufficient data for classification report"
        
    try:
        # zero_division=0 파라미터 추가
        f1 = f1_score(actual_directions, predicted_directions, average='macro', zero_division=0)
        report = classification_report(actual_directions, predicted_directions, 
                                    digits=2, zero_division=0)
    except Exception as e:
        logger.error(f"Error in calculating F1 score: {str(e)}")
        return 0.0, "Error in calculation"
        
    return f1, report

def calculate_direction_accuracy(actual, predicted):
    """등락 방향 예측의 정확도 계산"""
    if len(actual) <= 1:
        return 0.0
        
    try:
        actual_directions = np.sign(np.diff(actual))
        predicted_directions = np.sign(np.diff(predicted))
        
        correct_predictions = (actual_directions == predicted_directions).sum()
        total_predictions = len(actual_directions)
        
        accuracy = (correct_predictions / total_predictions) * 100
        return accuracy
    except Exception as e:
        logger.error(f"Error in calculating direction accuracy: {str(e)}")
        return 0.0
    
def calculate_direction_weighted_score(actual, predicted):
    """변화율 기반의 가중 점수 계산"""
    if len(actual) <= 1:
        return 0.0, 1.0
        
    try:
        actual_changes = 100 * np.diff(actual) / actual[:-1]
        predicted_changes = 100 * np.diff(predicted) / predicted[:-1]

        def assign_class(change):
            if change > 6:
                return 1
            elif 4 < change <= 6:
                return 2
            elif 2 < change <= 4:
                return 3
            elif -2 <= change <= 2:
                return 4
            elif -4 <= change < -2:
                return 5
            elif -6 <= change < -4:
                return 6
            else:
                return 7

        actual_classes = np.array([assign_class(x) for x in actual_changes])
        predicted_classes = np.array([assign_class(x) for x in predicted_changes])

        score = 0
        for ac, pc in zip(actual_classes, predicted_classes):
            diff = abs(ac - pc)
            score += max(0, 3 - diff)

        max_score = 3 * len(actual_classes)
        return score, max_score
    except Exception as e:
        logger.error(f"Error in calculating weighted score: {str(e)}")
        return 0.0, 1.0

def calculate_mape(actual, predicted):
    """MAPE 계산 함수"""
    try:
        if len(actual) == 0:
            return 0.0
        # inf 방지를 위해 0이 아닌 값만 사용
        mask = actual != 0
        if not any(mask):
            return 0.0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    except Exception as e:
        logger.error(f"Error in MAPE calculation: {str(e)}")
        return 0.0

def calculate_moving_averages_with_history(predictions, historical_data, target_col='MOPJ', windows=[5, 10, 23]):
    """예측 데이터와 과거 데이터를 모두 활용한 이동평균 계산"""
    try:
        results = {}
        
        # 예측 데이터를 DataFrame으로 변환 및 정렬
        pred_df = pd.DataFrame(predictions) if not isinstance(predictions, pd.DataFrame) else predictions.copy()
        pred_df['Date'] = pd.to_datetime(pred_df['Date'])
        pred_df = pred_df.sort_values('Date')
        
        # 예측 시작일 확인
        prediction_start_date = pred_df['Date'].min()
        
        # 과거 데이터에서 타겟 열 추출 (예측 시작일 이전)
        historical_series = pd.Series(
            data=historical_data.loc[historical_data.index < prediction_start_date, target_col],
            index=historical_data.loc[historical_data.index < prediction_start_date].index
        )
        
        # 최근 30일만 사용 (이동평균 계산에 충분)
        historical_series = historical_series.sort_index().tail(30)
        
        # 예측 데이터에서 시리즈 생성
        prediction_series = pd.Series(
            data=pred_df['Prediction'].values,
            index=pred_df['Date']
        )
        
        # 과거와 예측 데이터 결합
        combined_series = pd.concat([historical_series, prediction_series])
        combined_series = combined_series[~combined_series.index.duplicated(keep='last')]
        combined_series = combined_series.sort_index()
        
        logger.info(f"Combined series for MA: {len(combined_series)} data points "
                   f"({len(historical_series)} historical, {len(prediction_series)} predicted)")
        
        # 각 윈도우 크기별 이동평균 계산
        for window in windows:
            # 전체 데이터에 대해 이동평균 계산
            rolling_avg = combined_series.rolling(window=window, min_periods=1).mean()
            
            # 예측 기간에 해당하는 부분만 추출
            window_results = []
            
            for i, date in enumerate(pred_df['Date']):
                # 해당 날짜의 예측 및 실제값
                pred_value = pred_df['Prediction'].iloc[i]
                actual_value = pred_df['Actual'].iloc[i] if 'Actual' in pred_df.columns else None
                
                # 해당 날짜의 이동평균 값
                ma_value = rolling_avg.loc[date] if date in rolling_avg.index else None
                
                window_results.append({
                    'date': date,
                    'prediction': pred_value,
                    'actual': actual_value,
                    'ma': ma_value
                })
            
            results[f'ma{window}'] = window_results
        
        return results
    except Exception as e:
        logger.error(f"Error calculating moving averages with history: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

# 2. 여러 날짜에 대한 누적 예측을 수행하는 함수 추가
def run_accumulated_predictions(file_path, start_date, end_date=None):
    """
    시작 날짜부터 종료 날짜까지 각 날짜별로 예측을 수행하고 결과를 누적합니다.
    """
    global prediction_state

    try:
        # 상태 초기화
        prediction_state['is_predicting'] = True
        prediction_state['prediction_progress'] = 5
        prediction_state['error'] = None
        prediction_state['accumulated_predictions'] = []
        prediction_state['accumulated_metrics'] = {}
        prediction_state['prediction_dates'] = []
        # 누적 구간 점수를 저장할 딕셔너리 초기화
        accumulated_interval_scores = {}

        logger.info(f"Running accumulated predictions from {start_date} to {end_date}")

        # 입력 날짜를 datetime 객체로 변환
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is not None and isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # 데이터 로드
        df = load_data(file_path)
        prediction_state['current_data'] = df
        prediction_state['prediction_progress'] = 10

        # 종료 날짜가 지정되지 않으면 데이터의 마지막 날짜 사용
        if end_date is None:
            end_date = df.index.max()

        # 사용 가능한 날짜 추출 후 정렬
        available_dates = [date for date in df.index if start_date <= date <= end_date]
        available_dates.sort()
        logger.info(f"Filtered dates: {[format_date(d) for d in available_dates]}")
        if not available_dates:
            raise ValueError(f"지정된 기간 내에 사용 가능한 날짜가 없습니다: {start_date} ~ {end_date}")

        total_dates = len(available_dates)
        logger.info(f"Accumulated prediction: {total_dates} dates from {start_date} to {end_date}")

        # 누적 성능 지표 초기화
        accumulated_metrics = {
            'f1': 0.0,
            'accuracy': 0.0,
            'mape': 0.0,
            'weighted_score': 0.0,
            'total_predictions': 0
        }

        # 누적 예측 결과를 저장할 리스트를 초기화합니다.
        all_predictions = []

        # 각 날짜별 예측 수행
        for i, current_date in enumerate(available_dates):
            logger.info(f"Running prediction for date {i+1}/{total_dates}: {current_date}")
            try:
                results = generate_predictions(df, current_date)
                metrics = results['metrics']
                # 누적 성능 지표 업데이트
                accumulated_metrics['f1'] += metrics['f1']
                accumulated_metrics['accuracy'] += metrics['accuracy']
                accumulated_metrics['mape'] += metrics['mape']
                accumulated_metrics['weighted_score'] += metrics['weighted_score']
                accumulated_metrics['total_predictions'] += 1

                # 구간 점수 누적 처리
                # results['interval_scores']가 딕셔너리 형태라고 가정합니다.
                for interval in results['interval_scores'].values():
                    # Guard: interval이 정의되어 있고 'days' 프로퍼티가 존재하는지 확인
                    if not interval or 'days' not in interval or interval['days'] is None:
                        continue
                    interval_key = f"{format_date(interval['start_date'])}-{format_date(interval['end_date'])}"
                    if interval_key in accumulated_interval_scores:
                        accumulated_interval_scores[interval_key]['score'] += interval['score']
                        accumulated_interval_scores[interval_key]['count'] += 1
                        accumulated_interval_scores[interval_key]['avg_price'] = (
                            (accumulated_interval_scores[interval_key]['avg_price'] *
                             (accumulated_interval_scores[interval_key]['count'] - 1) +
                             interval['avg_price']) / accumulated_interval_scores[interval_key]['count']
                        )
                    else:
                        accumulated_interval_scores[interval_key] = interval.copy()
                        accumulated_interval_scores[interval_key]['count'] = 1

                date_result = {
                    'date': format_date(current_date),
                    'predictions': results['predictions'],
                    'metrics': metrics,
                    'interval_scores': results['interval_scores'],
                    'original_interval_scores': results['interval_scores']
                }
                all_predictions.append(date_result)
                prediction_state['prediction_progress'] = 10 + int(90 * (i + 1) / total_dates)
            except Exception as e:
                logger.error(f"Error in prediction for date {current_date}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        # 평균 성능 지표 계산
        if accumulated_metrics['total_predictions'] > 0:
            count = accumulated_metrics['total_predictions']
            accumulated_metrics['f1'] /= count
            accumulated_metrics['accuracy'] /= count
            accumulated_metrics['mape'] /= count
            accumulated_metrics['weighted_score'] /= count

        # accumulated_interval_scores 딕셔너리의 값들을 유효한 항목만 리스트로 변환 및 정렬
        accumulated_scores_list = [
            v for v in accumulated_interval_scores.values()
            if v is not None and isinstance(v, dict) and 'days' in v and v['days'] is not None
        ]
        accumulated_scores_list.sort(key=lambda x: x['score'], reverse=True)
        
        # 결과 저장
        prediction_state['accumulated_predictions'] = all_predictions
        prediction_state['accumulated_metrics'] = accumulated_metrics
        prediction_state['prediction_dates'] = [p['date'] for p in all_predictions]
        prediction_state['accumulated_interval_scores'] = accumulated_scores_list

        if all_predictions:
            latest = all_predictions[-1]
            prediction_state['latest_predictions'] = latest['predictions']
            prediction_state['current_date'] = latest['date']

        prediction_state['is_predicting'] = False
        prediction_state['prediction_progress'] = 100
        logger.info(f"Accumulated prediction completed for {len(all_predictions)} dates")
    except Exception as e:
        logger.error(f"Error in accumulated prediction: {str(e)}")
        logger.error(traceback.format_exc())
        prediction_state['error'] = str(e)
        prediction_state['is_predicting'] = False
        prediction_state['prediction_progress'] = 0

# 3. 백그라운드에서 누적 예측을 수행하는 함수
def background_accumulated_prediction(file_path, start_date, end_date=None):
    """백그라운드에서 누적 예측을 수행하는 함수"""
    thread = Thread(target=run_accumulated_predictions, args=(file_path, start_date, end_date))
    thread.daemon = True
    thread.start()
    return thread

# 6. 누적 결과 보고서 생성 함수
def generate_accumulated_report():
    """누적 예측 결과 보고서 생성"""
    global prediction_state
    
    if not prediction_state['accumulated_predictions']:
        return None
    
    try:
        metrics = prediction_state['accumulated_metrics']
        all_preds = prediction_state['accumulated_predictions']
        
        # 보고서 파일 이름 생성
        start_date = all_preds[0]['date']
        end_date = all_preds[-1]['date']
        report_filename = os.path.join(REPORT_DIR, f"accumulated_report_{start_date}_to_{end_date}.txt")
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"===== Accumulated Prediction Report =====\n")
            f.write(f"Period: {start_date} to {end_date}\n")
            f.write(f"Total Predictions: {metrics['total_predictions']}\n\n")
            
            # 누적 성능 지표
            f.write("Average Performance Metrics:\n")
            f.write(f"- F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"- Direction Accuracy: {metrics['accuracy']:.2f}%\n")
            f.write(f"- MAPE: {metrics['mape']:.2f}%\n")
            f.write(f"- Weighted Score: {metrics['weighted_score']:.2f}%\n\n")
            
            # 날짜별 상세 정보
            f.write("Performance By Date:\n")
            for pred in all_preds:
                date = pred['date']
                m = pred['metrics']
                f.write(f"\n* {date}:\n")
                f.write(f"  - F1 Score: {m['f1']:.4f}\n")
                f.write(f"  - Accuracy: {m['accuracy']:.2f}%\n")
                f.write(f"  - MAPE: {m['mape']:.2f}%\n")
                f.write(f"  - Weighted Score: {m['weighted_score']:.2f}%\n")
                
                # 구매 구간 정보
                if pred['interval_scores']:
                    best_interval = decide_purchase_interval(pred['interval_scores'])
                    f.write("Best Purchase Interval:\n")
                    f.write(f"- Start Date: {best_interval['start_date']}\n")
                    f.write(f"- End Date: {best_interval['end_date']}\n")
                    f.write(f"- Duration: {best_interval['days']} days\n")
                    f.write(f"- Average Price: {best_interval['avg_price']:.2f}\n")
                    f.write(f"- Score: {best_interval['score']}\n")
                    f.write(f"- Selection Reason: {best_interval.get('selection_reason', '')}\n\n")
        
        return report_filename
    
    except Exception as e:
        logger.error(f"Error generating accumulated report: {str(e)}")
        return None

# 9. 누적 예측 결과 시각화 함수
def visualize_accumulated_metrics():
    """누적 예측 결과 시각화"""
    global prediction_state
    
    if not prediction_state['accumulated_predictions']:
        return None, None
    
    try:
        # 데이터 준비
        dates = []
        f1_scores = []
        accuracies = []
        mapes = []
        weighted_scores = []
        
        for pred in prediction_state['accumulated_predictions']:
            dates.append(pred['date'])
            m = pred['metrics']
            f1_scores.append(m['f1'])
            accuracies.append(m['accuracy'])
            mapes.append(m['mape'])
            weighted_scores.append(m['weighted_score'])
        
        # 날짜를 datetime으로 변환
        dates = [pd.to_datetime(d) for d in dates]
        
        # 그래프 생성
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Accumulated Prediction Metrics', fontsize=16)
        
        # F1 Score
        axs[0, 0].plot(dates, f1_scores, marker='o', color='blue')
        axs[0, 0].set_title('F1 Score')
        axs[0, 0].set_ylim(0, 1)
        axs[0, 0].grid(True)
        plt.setp(axs[0, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Accuracy
        axs[0, 1].plot(dates, accuracies, marker='o', color='green')
        axs[0, 1].set_title('Direction Accuracy (%)')
        axs[0, 1].set_ylim(0, 100)
        axs[0, 1].grid(True)
        plt.setp(axs[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # MAPE
        axs[1, 0].plot(dates, mapes, marker='o', color='red')
        axs[1, 0].set_title('MAPE (%)')
        axs[1, 0].set_ylim(0, max(mapes) * 1.2)
        axs[1, 0].grid(True)
        plt.setp(axs[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Weighted Score
        axs[1, 1].plot(dates, weighted_scores, marker='o', color='purple')
        axs[1, 1].set_title('Weighted Score (%)')
        axs[1, 1].set_ylim(0, 100)
        axs[1, 1].grid(True)
        plt.setp(axs[1, 1].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # 이미지 저장
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300)
        img_buf.seek(0)
        
        # Base64로 인코딩
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        
        # 파일로 저장
        filename = os.path.join(PLOT_DIR, 'accumulated_metrics.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename, img_str
        
    except Exception as e:
        logger.error(f"Error visualizing accumulated metrics: {str(e)}")
        return None, None

#######################################################################
# 예측 및 모델 학습 함수
#######################################################################

def prepare_data(train_data, val_data, sequence_length, predict_window, target_col_idx, augment=False):
    """학습 및 검증 데이터를 시퀀스 형태로 준비"""
    X_train, y_train, prev_train = [], [], []
    for i in range(len(train_data) - sequence_length - predict_window + 1):
        seq = train_data[i:i+sequence_length]
        target = train_data[i+sequence_length:i+sequence_length+predict_window, target_col_idx]
        prev_value = train_data[i+sequence_length-1, target_col_idx]
        X_train.append(seq)
        y_train.append(target)
        prev_train.append(prev_value)
        if augment:
            # 간단한 데이터 증강
            noise = np.random.normal(0, 0.001, seq.shape)
            aug_seq = seq + noise
            X_train.append(aug_seq)
            y_train.append(target)
            prev_train.append(prev_value)
    
    X_val, y_val, prev_val = [], [], []
    for i in range(len(val_data) - sequence_length - predict_window + 1):
        X_val.append(val_data[i:i+sequence_length])
        y_val.append(val_data[i+sequence_length:i+sequence_length+predict_window, target_col_idx])
        prev_val.append(val_data[i+sequence_length-1, target_col_idx])
    
    return map(np.array, [X_train, y_train, prev_train, X_val, y_val, prev_val])

def train_model(features, target_col, current_date, historical_data, device, params):
    """LSTM 모델 학습"""
    try:
        # 특성 이름 확인
        if target_col not in features:
            features.append(target_col)
        
        # 학습 데이터 준비 (현재 날짜까지)
        train_df = historical_data[features].copy()
        target_col_idx = train_df.columns.get_loc(target_col)
        
        # 스케일링
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_df)
        
        # 하이퍼파라미터
        sequence_length = params.get('sequence_length', 20)
        hidden_size = params.get('hidden_size', 128)
        num_layers = params.get('num_layers', 2)
        dropout = params.get('dropout', 0.2)
        learning_rate = params.get('learning_rate', 0.001)
        num_epochs = params.get('num_epochs', 100)
        batch_size = params.get('batch_size', 32)
        alpha = params.get('loss_alpha', 0.6)  # MSE 가중치
        beta = params.get('loss_beta', 0.2)    # Volatility 가중치
        gamma = params.get('loss_gamma', 0.15)  # 방향성 가중치
        delta = params.get('loss_delta', 0.05)  # 연속성 가중치
        patience = params.get('patience', 20)   # 조기 종료 인내
        predict_window = params.get('predict_window', 23)  # 예측 기간
        
        # 80/20 분할 (연대순)
        train_size = int(len(train_data) * 0.8)
        train_set = train_data[:train_size]
        val_set = train_data[train_size:]
        
        # 시퀀스 데이터 준비
        X_train, y_train, prev_train, X_val, y_val, prev_val = prepare_data(
            train_set, val_set, sequence_length, predict_window, target_col_idx
        )
        
        # 충분한 데이터가 있는지 확인
        if len(X_train) < batch_size:
            batch_size = max(1, len(X_train) // 2)
            logger.warning(f"Batch size reduced to {batch_size} due to limited data")
        
        if len(X_train) == 0 or len(X_val) == 0:
            raise ValueError("Insufficient data for training")
        
        # 데이터셋 및 로더 생성
        train_dataset = TimeSeriesDataset(X_train, y_train, device, prev_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g
        )
        
        val_dataset = TimeSeriesDataset(X_val, y_val, device, prev_val)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        # 모델 생성
        model = ImprovedLSTMPredictor(
            input_size=train_data.shape[1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            output_size=predict_window
        ).to(device)
        
        # 최적화기 및 손실 함수
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # 학습
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # 학습 모드
            model.train()
            train_loss = 0
            
            for X_batch, y_batch, prev_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(X_batch, prev_batch)
                loss, _ = composite_loss(
                    y_pred, y_batch, prev_batch,
                    alpha=alpha, beta=beta, gamma=gamma, delta=delta
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # 검증 모드
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for X_batch, y_batch, prev_batch in val_loader:
                    y_pred = model(X_batch, prev_batch)
                    loss, _ = composite_loss(
                        y_pred, y_batch, prev_batch,
                        alpha=alpha, beta=beta, gamma=gamma, delta=delta
                    )
                    val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # 모델 저장 (최적)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 조기 종료
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 최적 모델 복원
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        logger.info(f"Model training completed with best validation loss: {best_val_loss:.4f}")
        
        # 모델, 스케일러, 파라미터 반환
        return model, scaler, target_col_idx
    
    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        logger.error(traceback.format_exc())
        raise e
    
# generate_predictions 함수를 수정합니다.
# 'sequence_df' 변수 정의 문제를 해결합니다.

def generate_predictions(df, current_date, predict_window=23, features=None, target_col='MOPJ'):
    """예측 수행 함수"""
    try:
        # 디바이스 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # 현재 날짜가 문자열이면 datetime으로 변환
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        # 현재 날짜 검증
        if current_date not in df.index:
            closest_date = df.index[df.index <= current_date][-1]
            logger.warning(f"Current date {current_date} not found in dataframe. Using closest date: {closest_date}")
            current_date = closest_date
        
        # 반월 기간 계산
        semimonthly_period = get_semimonthly_period(current_date)
        next_semimonthly_period = get_next_semimonthly_period(current_date)
        
        logger.info(f"Current date: {current_date}")
        logger.info(f"Semimonthly period: {semimonthly_period}")
        logger.info(f"Next semimonthly period: {next_semimonthly_period}")
        
        # 23일치 예측을 위한 날짜 생성
        all_business_days = get_next_n_business_days(current_date, df, predict_window)
        
        # 다음 반월 기간의 날짜만 추출
        semimonthly_business_days, next_period = get_next_semimonthly_dates(current_date, df)
        
        if not all_business_days:
            raise ValueError(f"No future business days found after {current_date}")

        logger.info(f"Predicting for {len(all_business_days)} days")
        logger.info(f"Next semimonthly period has {len(semimonthly_business_days)} business days")
        
        # 특성 선택 (사용자 지정 특성이 없으면 그룹별 선택)
        historical_data = df[df.index <= current_date].copy()
        
        if features is None:
            # 상관관계 기반 특성 선택
            selected_features, _ = select_features_from_groups(
                historical_data, 
                variable_groups,
                target_col=target_col,
                vif_threshold=50.0,
                corr_threshold=0.8
            )
        else:
            selected_features = features
            
        # 타겟 컬럼이 없으면 추가
        if target_col not in selected_features:
            selected_features.append(target_col)
        
        logger.info(f"Selected features ({len(selected_features)}): {selected_features}")
        
        # 선택된 특성으로 스케일링
        scaled_data = StandardScaler().fit_transform(historical_data[selected_features])
        target_col_idx = selected_features.index(target_col)
        
        # 하이퍼파라미터 최적화 수행
        optimized_params = optimize_hyperparameters_semimonthly_kfold(
            train_data=scaled_data,
            input_size=len(selected_features),
            target_col_idx=target_col_idx,
            device=device,
            current_period=semimonthly_period,
            n_trials=30,
            k_folds=5,
            use_cache=True
        )
        
        logger.info(f"Optimized hyperparameters for {semimonthly_period}: {optimized_params}")
        
        # 모델 학습
        model, scaler, target_col_idx = train_model(
            selected_features,
            target_col,
            current_date,
            historical_data,
            device,
            optimized_params
        )
        
        # 예측 데이터 준비
        seq_len = optimized_params['sequence_length']
        current_idx = df.index.get_loc(current_date)
        start_idx = max(0, current_idx - seq_len + 1)
        sequence = df.iloc[start_idx:current_idx+1][selected_features].values
        sequence = scaler.transform(sequence)
        prev_value = sequence[-1, target_col_idx]
        
        # 예측 수행 부분
        predictions = []
        
        # 예측 시퀀스 준비
        with torch.no_grad():
            # 23영업일 전체에 대해 예측 수행
            max_pred_days = min(predict_window, len(all_business_days))
            current_sequence = sequence.copy()
            
            # 텐서로 변환
            X = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
            prev_tensor = torch.FloatTensor([prev_value]).to(device)
            
            # 전체 시퀀스 예측
            pred = model(X, prev_tensor).cpu().numpy()[0]
            
            # 각 날짜별 예측 생성 (23일 전체)
            for j, pred_date in enumerate(all_business_days[:max_pred_days]):
                # 스케일 역변환
                dummy_matrix = np.zeros((1, len(selected_features)))
                dummy_matrix[0, target_col_idx] = pred[j]
                pred_value = scaler.inverse_transform(dummy_matrix)[0, target_col_idx]
                
                # 실제 값 (있는 경우만)
                actual_value = None
                if pred_date in df.index:
                    actual_value = df.loc[pred_date, target_col]
                
                # 예측 결과 저장
                predictions.append({
                    'Date': pred_date,
                    'Prediction': float(pred_value),
                    'Actual': float(actual_value) if actual_value is not None else None,
                    'Prediction_From': current_date,
                    'SemimonthlyPeriod': semimonthly_period,
                    'NextSemimonthlyPeriod': next_semimonthly_period,
                    'is_synthetic': pred_date not in df.index
                })
        
        # 구간 평균 및 점수 계산은 다음 반월 기간의 날짜만 사용
        interval_averages, interval_scores, analysis_info = calculate_interval_averages_and_scores(
            predictions, 
            semimonthly_business_days  # 다음 반월 기간의 영업일만 사용
        )

        # 최종 구매 구간 결정
        best_interval = decide_purchase_interval(interval_scores)

        # 예측 결과를 DataFrame으로 변환 - 오류가 발생하는 부분을 여기서 수정
        sequence_df = pd.DataFrame(predictions)

        # 성능 메트릭 계산 - 실제값이 있는 경우에만
        if any(p['Actual'] is not None for p in predictions):
            start_day_value = df.loc[current_date, target_col]
            metrics = compute_performance_metrics(sequence_df, start_day_value)
        else:
            # 실제값이 없는 경우 기본 메트릭 사용
            metrics = {
                'f1': 0.0,
                'accuracy': 0.0, 
                'mape': 0.0,
                'weighted_score': 0.0,
                'cosine_similarity': None,
                'f1_report': "No actual data available for metrics calculation"
            }
        
        # 이동평균 계산
        ma_results = calculate_moving_averages_with_history(
            predictions, 
            historical_data,
            target_col=target_col
        )
        
        # 특성 중요도 분석
        attention_data = None
        try:
            # 어텐션 분석 위한 시퀀스 준비
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            prev_tensor = torch.FloatTensor([float(prev_value)]).to(device)
            
            # 어텐션 맵 시각화
            attention_file, attention_img, feature_importance = visualize_attention_weights(
                model, sequence_tensor, prev_tensor, current_date, selected_features
            )
            
            attention_data = {
                'image': attention_img,
                'file_path': attention_file,
                'feature_importance': feature_importance
            }
        except Exception as e:
            logger.error(f"Error in attention analysis: {str(e)}")
        
        # 예측 결과 시각화
        start_day_value = df.loc[current_date, target_col] if current_date in df.index else None
        
        if start_day_value is not None:
            basic_plot_file, basic_plot_img = plot_prediction_basic(
                sequence_df, 
                current_date, 
                start_day_value,
                metrics['f1'],
                metrics['accuracy'],
                metrics['mape'],
                metrics['weighted_score'],
                save_prefix=PLOT_DIR
            )
            
            ma_plot_file, ma_plot_img = plot_moving_average_analysis(
                ma_results, 
                current_date,
                save_prefix=MA_PLOT_DIR
            )
        else:
            logger.warning(f"Start day value not available for visualization")
            basic_plot_file, basic_plot_img = None, None
            ma_plot_file, ma_plot_img = None, None

        # 결과 리포트 생성
        report_filename = os.path.join(REPORT_DIR, f"prediction_report_{format_date(current_date, '%Y%m%d')}.txt")
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"===== Prediction Report for {format_date(current_date)} =====\n\n")
            
            # 기본 정보
            f.write(f"Current Date: {format_date(current_date)}\n")
            f.write(f"Semimonthly Period: {semimonthly_period}\n")
            f.write(f"Next Semimonthly Period: {next_semimonthly_period}\n\n")
            
            # 성능 지표
            f.write("Performance Metrics:\n")
            f.write(f"- F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"- Direction Accuracy: {metrics['accuracy']:.2f}%\n")
            f.write(f"- MAPE: {metrics['mape']:.2f}%\n")
            f.write(f"- Weighted Score: {metrics['weighted_score']:.2f}%\n")
            if metrics['cosine_similarity'] is not None:
                f.write(f"- Cosine Similarity: {metrics['cosine_similarity']:.4f}\n")
            f.write("\n")
            
            # 특성 정보
            f.write(f"Selected Features ({len(selected_features)}):\n")
            for feature in selected_features:
                f.write(f"- {feature}\n")
            f.write("\n")
            
            # 최적 구매 구간
            if best_interval:
                f.write("Best Purchase Interval:\n")
                f.write(f"- Start Date: {best_interval['start_date']}\n")
                f.write(f"- End Date: {best_interval['end_date']}\n")
                f.write(f"- Duration: {best_interval['days']} days\n")
                f.write(f"- Average Price: {best_interval['avg_price']:.2f}\n")
                f.write(f"- Score: {best_interval['score']}\n")
                f.write(f"- Selection Reason: {best_interval.get('selection_reason', '')}\n\n")
            
            # 예측 상세 정보
            f.write("Detailed Predictions:\n")
            for pred in predictions:
                f.write(f"- {pred['Date']}: Prediction={pred['Prediction']:.2f}")
                if pred['Actual'] is not None:
                    actual = pred['Actual']
                    error = abs(pred['Prediction'] - actual)
                    error_pct = (error / actual) * 100 if actual != 0 else 0
                    f.write(f", Actual={actual:.2f}, Error={error:.2f} ({error_pct:.2f}%)")
                f.write("\n")
        
        # 결과 반환
        return {
            'predictions': predictions,
            'interval_scores': interval_scores,
            'interval_averages': interval_averages,
            'best_interval': best_interval,
            'ma_results': ma_results,
            'metrics': metrics,
            'selected_features': selected_features,
            'attention_data': attention_data,
            'plots': {
                'basic_plot': {
                    'file': basic_plot_file,
                    'image': basic_plot_img
                },
                'ma_plot': {
                    'file': ma_plot_file,
                    'image': ma_plot_img
                }
            },
            'report_file': report_filename,
            'current_date': format_date(current_date),
            'semimonthly_period': semimonthly_period,
            'next_semimonthly_period': next_semimonthly_period
        }
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

#######################################################################
# 백그라운드 작업 처리
#######################################################################

def background_prediction(file_path, current_date):
    """백그라운드에서 예측 작업을 수행하는 함수"""
    global prediction_state
    
    try:
        prediction_state['is_predicting'] = True
        prediction_state['prediction_progress'] = 10
        prediction_state['error'] = None
        
        # 데이터 로드 및 전처리
        df = load_data(file_path)
        prediction_state['prediction_progress'] = 30
        prediction_state['current_data'] = df
        
        # 현재 날짜 설정 (파일에서 가장 최근 날짜 사용)
        if current_date is None:
            current_date = df.index.max()
        else:
            current_date = pd.to_datetime(current_date)
        
        prediction_state['prediction_progress'] = 50
        
        # 예측 실행
        results = generate_predictions(df, current_date)
        prediction_state['prediction_progress'] = 80
        
        # 결과 저장
        prediction_state['latest_predictions'] = results['predictions']
        prediction_state['latest_interval_scores'] = results['interval_scores']
        prediction_state['latest_ma_results'] = results['ma_results']
        prediction_state['latest_attention_data'] = results['attention_data']
        prediction_state['current_date'] = results['current_date']
        prediction_state['selected_features'] = results['selected_features']
        prediction_state['feature_importance'] = results['attention_data']['feature_importance'] if results['attention_data'] else None
        prediction_state['semimonthly_period'] = results['semimonthly_period']
        prediction_state['next_semimonthly_period'] = results['next_semimonthly_period']
        
        prediction_state['prediction_progress'] = 100
        prediction_state['is_predicting'] = False
        logger.info("Prediction completed successfully")
        
    except Exception as e:
        logger.error(f"Error in background prediction: {str(e)}")
        logger.error(traceback.format_exc())
        prediction_state['error'] = str(e)
        prediction_state['is_predicting'] = False
        prediction_state['prediction_progress'] = 0

#######################################################################
# API 엔드포인트
#######################################################################

@app.route('/api/health', methods=['GET'])
def health_check():
    """서버 상태 확인 API"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/test', methods=['GET'])
def test_api():
    """API 연결 테스트"""
    return jsonify({
        'status': 'ok',
        'message': 'API is working!',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """CSV 파일 업로드 API"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and file.filename.endswith('.csv'):
        try:
            # 임시 파일명 생성
            filename = secure_filename(f"data_{int(time.time())}.csv")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # 파일 저장
            file.save(filepath)
            
            return jsonify({
                'success': True,
                'filepath': filepath,
                'filename': filename
            })
        except Exception as e:
            logger.error(f"Error during file upload: {str(e)}")
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Only CSV files are allowed'}), 400

@app.route('/api/holidays', methods=['GET'])
def get_holidays():
    """휴일 목록 조회 API"""
    return jsonify({
        'success': True,
        'holidays': list(holidays),
        'count': len(holidays)
    })

@app.route('/api/holidays/upload', methods=['POST'])
def upload_holidays():
    """휴일 목록 파일 업로드 API"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        try:
            # 임시 파일명 생성
            filename = secure_filename(f"holidays_{int(time.time())}{os.path.splitext(file.filename)[1]}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # 파일 저장
            file.save(filepath)
            
            # 휴일 정보 업데이트
            new_holidays = update_holidays(filepath)
            
            # 원본 파일을 모델 디렉토리로 복사 (standard location)
            permanent_path = os.path.join('models', 'holidays' + os.path.splitext(file.filename)[1])
            shutil.copy2(filepath, permanent_path)
            
            return jsonify({
                'success': True,
                'message': f'Successfully uploaded and loaded {len(new_holidays)} holidays',
                'filepath': permanent_path,
                'filename': os.path.basename(permanent_path),
                'holidays': list(new_holidays)
            })
        except Exception as e:
            logger.error(f"Error during holiday file upload: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Only CSV and Excel files are allowed'}), 400

@app.route('/api/holidays/reload', methods=['POST'])
def reload_holidays():
    """휴일 목록 재로드 API"""
    filepath = request.json.get('filepath')
    
    # 기본 파일 또는 지정된 파일로부터 재로드
    new_holidays = update_holidays(filepath)
    
    return jsonify({
        'success': True,
        'message': f'Successfully reloaded {len(new_holidays)} holidays',
        'holidays': list(new_holidays)
    })

@app.route('/api/file/metadata', methods=['GET'])
def get_file_metadata():
    """파일 메타데이터 조회 API"""
    filepath = request.args.get('filepath')
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # 기본 정보만 읽어서 반환
        df = pd.read_csv(filepath, nrows=5)  # 처음 5행만 읽기
        columns = df.columns.tolist()
        latest_date = None
        
        if 'Date' in df.columns:
            # 날짜 정보를 별도로 읽어서 최신 날짜 확인
            dates_df = pd.read_csv(filepath, usecols=['Date'])
            dates_df['Date'] = pd.to_datetime(dates_df['Date'])
            latest_date = dates_df['Date'].max().strftime('%Y-%m-%d')
        
        return jsonify({
            'success': True,
            'rows': len(df),
            'columns': columns,
            'latest_date': latest_date,
            'sample': df.head().to_dict(orient='records')
        })
    except Exception as e:
        logger.error(f"Error reading file metadata: {str(e)}")
        return jsonify({'error': f'Error reading file: {str(e)}'}), 500
    
@app.route('/api/data/dates', methods=['GET'])
def get_available_dates():
    filepath = request.args.get('filepath')
    days_limit = int(request.args.get('limit', 90))  # 기본값 90일로 확장, 쿼리 파라미터로 조절 가능
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 날짜 수 제한 확장
        dates = df['Date'].sort_values(ascending=False).head(days_limit).dt.strftime('%Y-%m-%d').tolist()
        
        return jsonify({
            'success': True,
            'dates': dates,
            'latest_date': dates[0] if dates else None
        })
    except Exception as e:
        logger.error(f"Error reading dates: {str(e)}")
        return jsonify({'error': f'Error reading dates: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def start_prediction():
    """예측 시작 API"""
    global prediction_state
    
    # 이미 예측 중인지 확인
    if prediction_state['is_predicting']:
        return jsonify({
            'success': False,
            'error': 'Prediction already in progress',
            'progress': prediction_state['prediction_progress']
        }), 409
    
    data = request.json
    filepath = data.get('filepath')
    current_date = data.get('date')  # 특정 날짜 지정 가능
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400
    
    # 백그라운드에서 예측 실행
    thread = Thread(target=background_prediction, args=(filepath, current_date))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Prediction started',
        'status_url': '/api/predict/status'
    })

@app.route('/api/predict/status', methods=['GET'])
def prediction_status():
    """예측 상태 확인 API"""
    global prediction_state
    
    status = {
        'is_predicting': prediction_state['is_predicting'],
        'progress': prediction_state['prediction_progress'],
        'error': prediction_state['error']
    }
    
    # 예측이 완료된 경우 날짜 정보도 반환
    if not prediction_state['is_predicting'] and prediction_state['current_date']:
        status['current_date'] = prediction_state['current_date']
    
    return jsonify(status)

@app.route('/api/results', methods=['GET'])
def get_prediction_results():
    """예측 결과 조회 API"""
    global prediction_state
    
    # 예측 중인 경우
    if prediction_state['is_predicting']:
        return jsonify({
            'success': False,
            'error': 'Prediction in progress',
            'progress': prediction_state['prediction_progress']
        }), 409
    
    # 예측 결과가 없는 경우
    if prediction_state['latest_predictions'] is None:
        return jsonify({'error': 'No prediction results available'}), 404
    
    # 디버깅 로그 추가
    logger.info(f"Returning prediction results: {len(prediction_state['latest_predictions'])} predictions")
    
    # 모든 결과 반환
    return jsonify({
        'success': True,
        'current_date': prediction_state['current_date'],
        'predictions': prediction_state['latest_predictions'],
        'interval_scores': prediction_state['latest_interval_scores'],
        'ma_results': prediction_state['latest_ma_results'],
        'attention_data': prediction_state['latest_attention_data'],
        'selected_features': prediction_state['selected_features'],
        'feature_importance': prediction_state['feature_importance'],
        'semimonthly_period': prediction_state['semimonthly_period'],
        'next_semimonthly_period': prediction_state['next_semimonthly_period']
    })

@app.route('/api/results/predictions', methods=['GET'])
def get_predictions_only():
    """예측 값만 조회 API"""
    global prediction_state
    
    if prediction_state['latest_predictions'] is None:
        return jsonify({'error': 'No prediction results available'}), 404
    
    return jsonify({
        'success': True,
        'current_date': prediction_state['current_date'],
        'predictions': prediction_state['latest_predictions']
    })

@app.route('/api/results/interval-scores', methods=['GET'])
def get_interval_scores():
    """구간 점수 조회 API"""
    global prediction_state
    
    if prediction_state['latest_interval_scores'] is None:
        return jsonify({'error': 'No interval scores available'}), 404
    
    # prediction_state['latest_interval_scores']가 dict인 경우 값을 배열로 변환,
    # 이미 배열이면 그대로 사용
    if isinstance(prediction_state['latest_interval_scores'], dict):
        interval_scores = list(prediction_state['latest_interval_scores'].values())
    else:
        interval_scores = prediction_state['latest_interval_scores']
    
    return jsonify({
        'success': True,
        'current_date': prediction_state['current_date'],
        'interval_scores': interval_scores
    })

@app.route('/api/results/moving-averages', methods=['GET'])
def get_moving_averages():
    """이동평균 조회 API"""
    global prediction_state
    
    if prediction_state['latest_ma_results'] is None:
        return jsonify({'error': 'No moving average results available'}), 404
    
    return jsonify({
        'success': True,
        'current_date': prediction_state['current_date'],
        'ma_results': prediction_state['latest_ma_results']
    })

@app.route('/api/results/attention-map', methods=['GET'])
def get_attention_map():
    """어텐션 맵 조회 API"""
    global prediction_state
    
    if prediction_state['latest_attention_data'] is None:
        return jsonify({'error': 'No attention map available'}), 404
    
    return jsonify({
        'success': True,
        'current_date': prediction_state['current_date'],
        'attention_data': prediction_state['latest_attention_data']
    })

@app.route('/api/features', methods=['GET'])
def get_features():
    """선택된 특성 조회 API"""
    global prediction_state
    
    if prediction_state['selected_features'] is None:
        return jsonify({'error': 'No feature information available'}), 404
    
    return jsonify({
        'success': True,
        'current_date': prediction_state['current_date'],
        'selected_features': prediction_state['selected_features'],
        'feature_importance': prediction_state['feature_importance']
    })

# 정적 파일 제공
@app.route('/static/<path:path>')
def serve_static(path):
    return send_file(os.path.join('static', path))

# 기본 라우트
@app.route('/')
def index():
    return jsonify({
        'app': 'MOPJ Prediction API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': [
            '/api/health',
            '/api/upload',
            '/api/file/metadata',
            '/api/data/dates',
            '/api/predict',
            '/api/predict/status',
            '/api/results',
            '/api/results/predictions',
            '/api/results/interval-scores',
            '/api/results/moving-averages',
            '/api/results/attention-map',
            '/api/features'
        ]
    })

# 4. API 엔드포인트 추가 - 누적 예측 시작
@app.route('/api/predict/accumulated', methods=['POST'])
def start_accumulated_prediction():
    """여러 날짜에 대한 누적 예측 시작 API"""
    global prediction_state
    
    # 이미 예측 중인지 확인
    if prediction_state['is_predicting']:
        return jsonify({
            'success': False,
            'error': 'Prediction already in progress',
            'progress': prediction_state['prediction_progress']
        }), 409
    
    data = request.json
    filepath = data.get('filepath')
    start_date = data.get('start_date')  # 시작 날짜 (필수)
    end_date = data.get('end_date')      # 종료 날짜 (선택)
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400
    
    if not start_date:
        return jsonify({'error': 'Start date is required'}), 400
    
    # 백그라운드에서 누적 예측 실행
    background_accumulated_prediction(filepath, start_date, end_date)
    
    return jsonify({
        'success': True,
        'message': 'Accumulated prediction started',
        'status_url': '/api/predict/status'
    })

# 5. API 엔드포인트 추가 - 누적 예측 결과 조회
@app.route('/api/results/accumulated', methods=['GET'])
def get_accumulated_results():
    global prediction_state
    if prediction_state['is_predicting']:
        return jsonify({
            'success': False,
            'error': 'Prediction in progress',
            'progress': prediction_state['prediction_progress']
        }), 409

    if not prediction_state['accumulated_predictions']:
        return jsonify({'error': 'No accumulated prediction results available'}), 404

    # 데이터 안전성 검사 및 필터링 추가
    # accumulated_interval_scores가 None이거나 days 속성이 없는 항목은 필터링
    safe_interval_scores = []
    if prediction_state.get('accumulated_interval_scores'):
        safe_interval_scores = [
            item for item in prediction_state['accumulated_interval_scores'] 
            if item is not None and isinstance(item, dict) and 'days' in item and item['days'] is not None
        ]
    
    return jsonify({
        'success': True,
        'prediction_dates': prediction_state['prediction_dates'],
        'accumulated_metrics': prediction_state['accumulated_metrics'],
        'predictions': prediction_state['accumulated_predictions'],
        'accumulated_interval_scores': safe_interval_scores  # 필터링된 데이터 사용
    })

@app.route('/api/results/accumulated/interval-scores', methods=['GET'])
def get_accumulated_interval_scores():
    global prediction_state
    scores = prediction_state.get('accumulated_interval_scores', [])
    
    # 'days' 속성이 없는 항목 필터링
    safe_scores = [
        item for item in scores 
        if item is not None and isinstance(item, dict) and 'days' in item and item['days'] is not None
    ]
    
    return jsonify(safe_scores)

# 7. 누적 보고서 API 엔드포인트
@app.route('/api/results/accumulated/report', methods=['GET'])
def get_accumulated_report():
    """누적 예측 결과 보고서 생성 및 다운로드 API"""
    global prediction_state
    
    # 예측 결과가 없는 경우
    if not prediction_state['accumulated_predictions']:
        return jsonify({'error': 'No accumulated prediction results available'}), 404
    
    report_file = generate_accumulated_report()
    if not report_file:
        return jsonify({'error': 'Failed to generate report'}), 500
    
    return send_file(report_file, as_attachment=True)

# 8. API 엔드포인트 추가 - 특정 날짜 예측 결과 조회
@app.route('/api/results/accumulated/<date>', methods=['GET'])
def get_accumulated_result_by_date(date):
    """특정 날짜의 누적 예측 결과 조회 API"""
    global prediction_state
    
    if not prediction_state['accumulated_predictions']:
        return jsonify({'error': 'No accumulated prediction results available'}), 404
    
    # 해당 날짜의 예측 결과 찾기
    for pred in prediction_state['accumulated_predictions']:
        if pred['date'] == date:
            return jsonify({
                'success': True,
                'date': date,
                'predictions': pred['predictions'],
                'metrics': pred['metrics'],
                'interval_scores': pred['interval_scores']
            })
    
    return jsonify({'error': f'No prediction results for date {date}'}), 404

# 10. 누적 지표 시각화 API 엔드포인트
@app.route('/api/results/accumulated/visualization', methods=['GET'])
def get_accumulated_visualization():
    """누적 예측 지표 시각화 API"""
    global prediction_state
    
    if not prediction_state['accumulated_predictions']:
        return jsonify({'error': 'No accumulated prediction results available'}), 404
    
    filename, img_str = visualize_accumulated_metrics()
    if not filename:
        return jsonify({'error': 'Failed to generate visualization'}), 500
    
    return jsonify({
        'success': True,
        'file_path': filename,
        'image': img_str
    })

# 메인 실행 부분
if __name__ == '__main__':
    # 필요한 패키지 설치 확인
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna 패키지가 설치되어 있지 않습니다. 하이퍼파라미터 최적화를 위해 설치가 필요합니다.")
        logger.warning("pip install optuna 명령으로 설치할 수 있습니다.")
    
    # 모델 및 캐시 디렉토리 생성
    os.makedirs('models', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
