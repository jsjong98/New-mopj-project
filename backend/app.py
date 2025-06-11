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
import csv
from pathlib import Path

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

# 디렉토리 설정 - 파일별 캐시 시스템
UPLOAD_FOLDER = 'uploads'
HOLIDAY_DIR = 'holidays'
CACHE_ROOT_DIR = 'cache'  # 🔑 새로운 파일별 캐시 루트

# 기본 디렉토리 생성 (최소한만 유지)
for d in [UPLOAD_FOLDER, CACHE_ROOT_DIR]:
    os.makedirs(d, exist_ok=True)

def get_file_cache_dirs(file_path=None):
    """
    파일별 캐시 디렉토리 구조를 반환하는 함수
    🎯 각 파일마다 독립적인 모델, 예측, 시각화 캐시 제공
    """
    try:
        if not file_path:
            file_path = prediction_state.get('current_file', None)
        
        # Debug: file cache directory setup
        
        if not file_path:
            logger.warning(f"⚠️ No file path provided and no current_file in prediction_state")
            # 기본 캐시 디렉토리 반환 (파일별 캐시 없이)
            default_cache_root = Path(CACHE_ROOT_DIR) / 'default'
            dirs = {
                'root': default_cache_root,
                'models': default_cache_root / 'models',
                'predictions': default_cache_root / 'predictions',
                'plots': default_cache_root / 'static' / 'plots',
                'ma_plots': default_cache_root / 'static' / 'ma_plots',
                'accumulated': default_cache_root / 'accumulated'
            }
            
            # Create default directories
            for name, dir_path in dirs.items():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.error(f"❌ Failed to create default {name} directory {dir_path}: {str(e)}")
            
            logger.warning(f"⚠️ Using default cache directory")
            return dirs
        
        if not os.path.exists(file_path):
            logger.error(f"❌ File does not exist: {file_path}")
            raise ValueError(f"File does not exist: {file_path}")
        
        # Generate file cache directory
        file_content_hash = get_data_content_hash(file_path)
        
        if not file_content_hash:
            logger.error(f"❌ Failed to get content hash for file: {file_path}")
            raise ValueError(f"Unable to generate content hash for file: {file_path}")
        
        file_name = Path(file_path).stem
        file_dir_name = f"{file_content_hash[:12]}_{file_name}"
        file_cache_root = Path(CACHE_ROOT_DIR) / file_dir_name
        
        dirs = {
            'root': file_cache_root,
            'models': file_cache_root / 'models',
            'predictions': file_cache_root / 'predictions',
            'plots': file_cache_root / 'static' / 'plots',
            'ma_plots': file_cache_root / 'static' / 'ma_plots',
            'accumulated': file_cache_root / 'accumulated'
        }
        
        # Create cache directories
        for name, dir_path in dirs.items():
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"❌ Failed to create {name} directory {dir_path}: {str(e)}")
        
        return dirs
        
    except Exception as e:
        logger.error(f"❌ Error in get_file_cache_dirs: {str(e)}")
        logger.error(traceback.format_exc())
        raise e  # 오류 발생 시 예외 전파

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',  # 더 간결한 로그 포맷
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask 설정
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=False)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 최대 파일 크기 32MB로 증가

# 전역 상태 변수에 새 필드 추가
prediction_state = {
    'current_data': None,
    'latest_predictions': None,
    'latest_interval_scores': None,
    'latest_attention_data': None,
    'latest_ma_results': None,
    'latest_plots': None,  # 추가
    'latest_metrics': None,  # 추가
    'current_date': None,
    'current_file': None,  # 추가: 현재 파일 경로
    'is_predicting': False,
    'prediction_progress': 0,
    'error': None,
    'selected_features': None,
    'feature_importance': None,
    'semimonthly_period': None,
    'next_semimonthly_period': None,
    'accumulated_predictions': [],
    'accumulated_metrics': {},
    'prediction_dates': [],
    'accumulated_consistency_scores': {},
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

# 🔧 스마트 파일 캐시 시스템 함수들
def calculate_file_hash(file_path, chunk_size=8192):
    """파일 내용의 SHA256 해시를 계산"""
    import hashlib
    
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"File hash calculation failed: {str(e)}")
        return None

def get_data_content_hash(file_path):
    """CSV 파일의 데이터 내용만으로 해시 생성 (날짜 순서 기준)"""
    import hashlib
    
    try:
        df = pd.read_csv(file_path)
        if 'Date' in df.columns:
            # 날짜를 기준으로 정렬하여 일관된 해시 생성
            df = df.sort_values('Date')
        
        # 데이터프레임의 내용을 문자열로 변환하여 해시 계산
        content_str = df.to_string()
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]  # 짧은 해시 사용
    except Exception as e:
        logger.error(f"Data content hash calculation failed: {str(e)}")
        return None

def check_data_extension(old_file_path, new_file_path):
    """
    새 파일이 기존 파일의 순차적 확장(기존 데이터 이후에만 새 행 추가)인지 엄격하게 확인
    
    ⚠️ 중요: 다음 경우만 확장으로 인정:
    1. 기존 데이터와 정확히 동일한 부분이 있음
    2. 새 데이터가 기존 데이터의 마지막 날짜 이후에만 추가됨
    3. 기존 데이터의 시작/중간 날짜가 변경되지 않음
    
    Returns:
    --------
    dict: {
        'is_extension': bool,
        'new_rows_count': int,
        'base_hash': str,  # 기존 데이터 부분의 해시
        'old_start_date': str,
        'old_end_date': str,
        'new_start_date': str,
        'new_end_date': str,
        'validation_details': dict
    }
    """
    try:
        old_df = pd.read_csv(old_file_path)
        new_df = pd.read_csv(new_file_path)
        
        # 날짜 컬럼이 있는지 확인
        if 'Date' not in old_df.columns or 'Date' not in new_df.columns:
            return {
                'is_extension': False, 
                'new_rows_count': 0,
                'validation_details': {'error': 'No Date column found'}
            }
        
        # 날짜로 정렬
        old_df = old_df.sort_values('Date').reset_index(drop=True)
        new_df = new_df.sort_values('Date').reset_index(drop=True)
        
        # 날짜를 datetime으로 변환
        old_df['Date'] = pd.to_datetime(old_df['Date'])
        new_df['Date'] = pd.to_datetime(new_df['Date'])
        
        # 기본 정보 추출
        old_start_date = old_df['Date'].iloc[0]
        old_end_date = old_df['Date'].iloc[-1]
        new_start_date = new_df['Date'].iloc[0]
        new_end_date = new_df['Date'].iloc[-1]
        
        logger.info(f"🔍 [EXTENSION_CHECK] Old data: {old_start_date.strftime('%Y-%m-%d')} ~ {old_end_date.strftime('%Y-%m-%d')} ({len(old_df)} rows)")
        logger.info(f"🔍 [EXTENSION_CHECK] New data: {new_start_date.strftime('%Y-%m-%d')} ~ {new_end_date.strftime('%Y-%m-%d')} ({len(new_df)} rows)")
        
        # ✅ 검증 1: 새 파일이 더 길어야 함
        if len(new_df) <= len(old_df):
            logger.info(f"❌ [EXTENSION_CHECK] New file is not longer ({len(new_df)} <= {len(old_df)})")
            return {
                'is_extension': False, 
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': 'New file is not longer than old file'}
            }
        
        # ✅ 검증 2: 새 데이터의 시작 날짜가 기존 데이터의 시작 날짜와 같거나 그 이후여야 함
        if new_start_date < old_start_date:
            logger.info(f"❌ [EXTENSION_CHECK] New data starts before old data ({new_start_date} < {old_start_date})")
            return {
                'is_extension': False,
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': 'New data contains dates before existing data start date'}
            }
        
        # ✅ 검증 3: 새 데이터의 마지막 날짜가 기존 데이터의 마지막 날짜 이후여야 함
        if new_end_date <= old_end_date:
            logger.info(f"❌ [EXTENSION_CHECK] New data doesn't extend beyond old data ({new_end_date} <= {old_end_date})")
            return {
                'is_extension': False,
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': 'New data does not extend beyond existing end date'}
            }
        
        # ✅ 검증 4: 기존 데이터의 모든 날짜가 새 데이터에 포함되어야 함
        old_dates = set(old_df['Date'].dt.strftime('%Y-%m-%d'))
        new_dates = set(new_df['Date'].dt.strftime('%Y-%m-%d'))
        
        missing_dates = old_dates - new_dates
        if missing_dates:
            logger.info(f"❌ [EXTENSION_CHECK] Some old dates are missing in new data: {missing_dates}")
            return {
                'is_extension': False,
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': f'Missing dates from old data: {list(missing_dates)}'}
            }
        
        # ✅ 검증 5: 컬럼이 동일해야 함
        if list(old_df.columns) != list(new_df.columns):
            logger.info(f"❌ [EXTENSION_CHECK] Column structure differs")
            return {
                'is_extension': False,
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': 'Column structure differs'}
            }
        
        # ✅ 검증 6: 기존 데이터 부분이 정확히 동일한지 확인 (날짜 기준으로 매칭)
        logger.info(f"🔍 [EXTENSION_CHECK] Comparing overlapping data...")
        
        # 기존 데이터의 각 날짜에 해당하는 새 데이터 행 찾기
        data_matches = True
        mismatch_details = []
        
        for idx, old_row in old_df.iterrows():
            old_date = old_row['Date']
            old_date_str = old_date.strftime('%Y-%m-%d')
            
            # 새 데이터에서 해당 날짜 찾기
            new_matching_rows = new_df[new_df['Date'] == old_date]
            
            if len(new_matching_rows) == 0:
                data_matches = False
                mismatch_details.append(f"Date {old_date_str} missing in new data")
                break
            elif len(new_matching_rows) > 1:
                data_matches = False
                mismatch_details.append(f"Duplicate date {old_date_str} in new data")
                break
            
            new_row = new_matching_rows.iloc[0]
            
            # 수치 컬럼 비교 (Date 제외)
            numeric_cols = old_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if not np.allclose([old_row[col]], [new_row[col]], rtol=1e-10, atol=1e-12, equal_nan=True):
                    data_matches = False
                    mismatch_details.append(f"Value mismatch on {old_date_str}, column {col}: {old_row[col]} != {new_row[col]}")
                    break
            
            if not data_matches:
                break
            
            # 문자열 컬럼 비교 (Date 제외)
            str_cols = old_df.select_dtypes(include=['object']).columns
            str_cols = [col for col in str_cols if col != 'Date']
            for col in str_cols:
                if old_row[col] != new_row[col]:
                    data_matches = False
                    mismatch_details.append(f"String mismatch on {old_date_str}, column {col}: '{old_row[col]}' != '{new_row[col]}'")
                    break
            
            if not data_matches:
                break
        
        if not data_matches:
            logger.info(f"❌ [EXTENSION_CHECK] Data content differs: {mismatch_details}")
            return {
                'is_extension': False,
                'new_rows_count': 0,
                'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                'validation_details': {'reason': 'Data content differs', 'details': mismatch_details}
            }
        
        # ✅ 검증 7: 새로 추가된 데이터가 기존 데이터의 마지막 날짜 이후에만 있는지 확인
        new_only_dates = new_dates - old_dates
        if new_only_dates:
            new_only_dates_dt = [pd.to_datetime(date) for date in new_only_dates]
            earliest_new_date = min(new_only_dates_dt)
            
            if earliest_new_date <= old_end_date:
                logger.info(f"❌ [EXTENSION_CHECK] New dates are not strictly after old end date: {earliest_new_date} <= {old_end_date}")
                return {
                    'is_extension': False,
                    'new_rows_count': 0,
                    'old_start_date': old_start_date.strftime('%Y-%m-%d'),
                    'old_end_date': old_end_date.strftime('%Y-%m-%d'),
                    'new_start_date': new_start_date.strftime('%Y-%m-%d'),
                    'new_end_date': new_end_date.strftime('%Y-%m-%d'),
                    'validation_details': {'reason': 'New dates are not strictly sequential after old end date'}
                }
        
        # ✅ 모든 검증 통과: 순차적 확장으로 인정
        new_rows_count = len(new_only_dates)
        base_hash = get_data_content_hash(old_file_path)
        
        logger.info(f"✅ [EXTENSION_CHECK] Valid sequential extension: +{new_rows_count} new dates after {old_end_date.strftime('%Y-%m-%d')}")
        
        return {
            'is_extension': True,
            'new_rows_count': new_rows_count,
            'base_hash': base_hash,
            'old_start_date': old_start_date.strftime('%Y-%m-%d'),
            'old_end_date': old_end_date.strftime('%Y-%m-%d'),
            'new_start_date': new_start_date.strftime('%Y-%m-%d'),
            'new_end_date': new_end_date.strftime('%Y-%m-%d'),
            'validation_details': {
                'reason': 'Valid sequential extension',
                'new_dates_added': sorted(list(new_only_dates))
            }
        }
        
    except Exception as e:
        logger.error(f"Data extension check failed: {str(e)}")
        return {
            'is_extension': False, 
            'new_rows_count': 0,
            'old_start_date': None,
            'old_end_date': None,
            'new_start_date': None,
            'new_end_date': None,
            'validation_details': {'error': str(e)}
        }

def find_compatible_cache_file(new_file_path):
    """
    새 파일과 호환되는 기존 캐시를 찾는 함수
    
    Returns:
    --------
    dict: {
        'found': bool,
        'cache_type': str,  # 'exact', 'extension', None
        'cache_file': str,
        'extension_info': dict
    }
    """
    try:
        # 새 파일의 데이터 해시
        new_hash = get_data_content_hash(new_file_path)
        if not new_hash:
            return {'found': False, 'cache_type': None}
        
        # uploads 폴더의 모든 CSV 파일 확인
        upload_dir = Path(UPLOAD_FOLDER)
        existing_files = list(upload_dir.glob('*.csv'))
        
        logger.info(f"🔍 [CACHE] Checking {len(existing_files)} existing files for cache compatibility")
        
        for existing_file in existing_files:
            if existing_file.name == os.path.basename(new_file_path):
                continue  # 자기 자신은 제외
            
            try:
                # 1. 정확한 매치 확인
                existing_hash = get_data_content_hash(str(existing_file))
                if existing_hash == new_hash:
                    logger.info(f"✅ [CACHE] Found exact match: {existing_file.name}")
                    return {
                        'found': True,
                        'cache_type': 'exact',
                        'cache_file': str(existing_file),
                        'extension_info': None
                    }
                
                # 2. 확장 파일인지 확인
                extension_info = check_data_extension(str(existing_file), new_file_path)
                if extension_info['is_extension']:
                    logger.info(f"📈 [CACHE] Found extension base: {existing_file.name} (+{extension_info['new_rows_count']} rows)")
                    return {
                        'found': True,
                        'cache_type': 'extension',
                        'cache_file': str(existing_file),
                        'extension_info': extension_info
                    }
                    
            except Exception as e:
                logger.warning(f"Error checking file {existing_file}: {str(e)}")
                continue
        
        logger.info("❌ [CACHE] No compatible cache found")
        return {'found': False, 'cache_type': None}
        
    except Exception as e:
        logger.error(f"Cache compatibility check failed: {str(e)}")
        return {'found': False, 'cache_type': None}

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
    # 기본 파일 경로 - holidays 폴더로 변경
    if filepath is None:
        holidays_dir = Path('holidays')
        holidays_dir.mkdir(exist_ok=True)
        filepath = str(holidays_dir / 'holidays.csv')
    
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

# 데이터에서 평일 빈 날짜를 휴일로 감지하는 함수
def detect_missing_weekdays_as_holidays(df, date_column='Date'):
    """
    데이터프레임에서 평일(월~금)인데 데이터가 없는 날짜들을 휴일로 감지하는 함수
    
    Args:
        df (pd.DataFrame): 데이터프레임
        date_column (str): 날짜 컬럼명
    
    Returns:
        set: 감지된 휴일 날짜 집합 (YYYY-MM-DD 형식)
    """
    if df.empty or date_column not in df.columns:
        return set()
    
    try:
        # 날짜 컬럼을 datetime으로 변환
        df_dates = pd.to_datetime(df[date_column]).dt.date
        date_set = set(df_dates)
        
        # 데이터 범위의 첫 날과 마지막 날
        start_date = min(df_dates)
        end_date = max(df_dates)
        
        # 전체 기간의 모든 평일 생성
        current_date = start_date
        missing_weekdays = set()
        
        while current_date <= end_date:
            # 평일인지 확인 (월요일=0, 일요일=6)
            if current_date.weekday() < 5:  # 월~금
                if current_date not in date_set:
                    missing_weekdays.add(current_date.strftime('%Y-%m-%d'))
            current_date += pd.Timedelta(days=1)
        
        logger.info(f"Detected {len(missing_weekdays)} missing weekdays as potential holidays")
        if missing_weekdays:
            logger.info(f"Missing weekdays sample: {list(missing_weekdays)[:10]}")
        
        return missing_weekdays
        
    except Exception as e:
        logger.error(f"Error detecting missing weekdays: {str(e)}")
        return set()

# 휴일 정보와 데이터 빈 날짜를 결합하는 함수
def get_combined_holidays(df=None, filepath=None):
    """
    휴일 파일의 휴일과 데이터에서 감지된 휴일을 결합하는 함수
    
    Args:
        df (pd.DataFrame): 데이터프레임 (빈 날짜 감지용)
        filepath (str): 휴일 파일 경로
    
    Returns:
        set: 결합된 휴일 날짜 집합
    """
    # 휴일 파일에서 휴일 로드
    file_holidays = load_holidays_from_file(filepath)
    
    # 데이터에서 빈 평일 감지
    data_holidays = set()
    if df is not None:
        data_holidays = detect_missing_weekdays_as_holidays(df)
    
    # 두 세트 결합
    combined_holidays = file_holidays.union(data_holidays)
    
    logger.info(f"Combined holidays: {len(file_holidays)} from file + {len(data_holidays)} from data = {len(combined_holidays)} total")
    
    return combined_holidays

# 휴일 정보 업데이트 함수
def update_holidays(filepath=None, df=None):
    """휴일 정보를 재로드하는 함수 (데이터 빈 날짜 포함)"""
    global holidays
    holidays = get_combined_holidays(df, filepath)
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
def get_next_semimonthly_dates(reference_date, original_df):
    """
    참조 날짜 기준으로 다음 반월 기간에 속하는 모든 영업일 목록을 반환하는 함수
    """
    # 다음 반월 기간 계산
    next_period = get_next_semimonthly_period(reference_date)
    
    logger.info(f"Calculating next semimonthly dates from reference: {format_date(reference_date)} → target period: {next_period}")
    
    # 반월 기간의 시작일과 종료일 계산
    start_date, end_date = get_semimonthly_date_range(next_period)
    
    logger.info(f"Target period date range: {format_date(start_date)} ~ {format_date(end_date)}")
    
    # 이 기간에 속하는 영업일(월~금, 휴일 제외) 선택
    business_days = []
    
    # 원본 데이터에서 찾기
    future_dates = original_df.index[original_df.index > reference_date]
    for date in future_dates:
        if start_date <= date <= end_date and date.weekday() < 5 and not is_holiday(date):
            business_days.append(date)
    
    # 원본 데이터에 없는 경우, 날짜 범위에서 직접 생성
    if len(business_days) == 0:
        logger.info(f"No business days found in original data for period {next_period}. Generating from date range.")
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5 and not is_holiday(current_date):
                business_days.append(current_date)
            current_date += pd.Timedelta(days=1)
    
    # 날짜가 없거나 부족하면 추가 로직
    min_required_days = 5
    if len(business_days) < min_required_days:
        logger.warning(f"Only {len(business_days)} business days found in period {next_period}. Creating synthetic dates.")
        
        if business_days:
            synthetic_date = business_days[-1] + pd.Timedelta(days=1)
        else:
            synthetic_date = max(reference_date, start_date) + pd.Timedelta(days=1)
        
        while len(business_days) < 15 and synthetic_date <= end_date:
            if synthetic_date.weekday() < 5 and not is_holiday(synthetic_date):
                business_days.append(synthetic_date)
            synthetic_date += pd.Timedelta(days=1)
        
        logger.info(f"Created synthetic dates. Total business days: {len(business_days)} for period {next_period}")
    
    business_days.sort()
    logger.info(f"Final business days for purchase interval: {len(business_days)} days in {next_period}")
    
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

def optimize_hyperparameters_semimonthly_kfold(train_data, input_size, target_col_idx, device, current_period, file_path=None, n_trials=30, k_folds=5, use_cache=True):
    """
    시계열 K-fold 교차 검증을 사용하여 반월별 데이터에 대한 하이퍼파라미터 최적화
    """
    logger.info(f"\n===== {current_period} 하이퍼파라미터 최적화 시작 (시계열 {k_folds}-fold 교차 검증) =====")
    
    # 캐시 파일 경로 - 파일별 캐시 디렉토리 사용
    file_cache_dir = get_file_cache_dirs(file_path)['models']
    cache_file = os.path.join(file_cache_dir, f"hyperparams_kfold_{current_period.replace('-', '_')}.json")
    logger.info(f"📁 하이퍼파라미터 캐시 파일: {cache_file}")
    
    # models 디렉토리 생성
    os.makedirs(file_cache_dir, exist_ok=True)
    
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
# 예측 결과 저장/로드 함수들
#######################################################################

def save_prediction_simple(prediction_results: dict, prediction_date):
    """리스트·딕트 어떤 구조든 저장 가능한 안전 버전 - 파일명 통일"""
    try:
        preds_root = prediction_results.get("predictions")

        # ── 첫 예측 레코드 추출 ─────────────────────────
        if isinstance(preds_root, dict) and preds_root:
            preds_seq = preds_root.get("future") or []
        else:                                   # list 혹은 None
            preds_seq = preds_root or prediction_results.get("predictions_flat", [])

        if not preds_seq:
            raise ValueError("prediction_results 안에 예측 데이터가 비어 있습니다.")

        first_rec = preds_seq[0]
        first_date = pd.to_datetime(first_rec.get("date") or first_rec.get("Date"))
        if pd.isna(first_date):
            raise ValueError("첫 예측 레코드에 날짜 정보가 없습니다.")

        # 🎯 파일별 캐시 디렉토리 사용
        cache_dirs = get_file_cache_dirs()
        file_predictions_dir = cache_dirs['predictions']
        
        # ✅ 파일 경로 설정 (파일별 디렉토리 내)
        json_path = file_predictions_dir / f"prediction_start_{first_date:%Y%m%d}.json"
        csv_path = file_predictions_dir / f"prediction_start_{first_date:%Y%m%d}.csv"
        meta_path = file_predictions_dir / f"prediction_start_{first_date:%Y%m%d}_meta.json"
        attention_path = file_predictions_dir / f"prediction_start_{first_date:%Y%m%d}_attention.json"
        
        logger.info(f"📁 Using file cache directory: {cache_dirs['root'].name}")
        logger.info(f"  📄 Predictions: {file_predictions_dir.name}")
        logger.info(f"  📄 CSV: {csv_path.name}")
        logger.info(f"  📄 Meta: {meta_path.name}")

        # ── validation 개수 계산 ──────────────────────
        if isinstance(preds_root, dict):
            validation_cnt = len(preds_root.get("validation", []))
        else:
            validation_cnt = 0

        # ── 메타 + 본문 구성 (파일 캐시 정보 포함) ──────────────────────────
        current_file_path = prediction_state.get('current_file', None)
        file_content_hash = get_data_content_hash(current_file_path) if current_file_path else None
        
        meta = {
            "prediction_start_date": first_date.strftime("%Y-%m-%d"),
            "data_end_date": str(prediction_date)[:10],
            "created_at": datetime.now().isoformat(),
            "semimonthly_period": prediction_results.get("semimonthly_period"),
            "next_semimonthly_period": prediction_results.get("next_semimonthly_period"),
            "selected_features": prediction_results.get("selected_features", []),
            "total_predictions": len(prediction_results.get("predictions_flat", preds_seq)),
            "validation_points": validation_cnt,
            "is_pure_future_prediction": prediction_results.get("summary", {}).get(
                "is_pure_future_prediction", validation_cnt == 0
            ),
            "metrics": prediction_results.get("metrics"),
            "interval_scores": prediction_results.get("interval_scores", {}),
            # 🔑 캐시 연동을 위한 파일 정보
            "file_path": current_file_path,
            "file_content_hash": file_content_hash
        }

        # ✅ CSV 파일 저장
        predictions_data = clean_predictions_data(
            prediction_results.get("predictions_flat", preds_seq)
        )
        
        if predictions_data:
            pred_df = pd.DataFrame(predictions_data)
            pred_df.to_csv(csv_path, index=False)
            logger.info(f"✅ CSV saved: {csv_path}")

        # ✅ 메타데이터 저장
        with open(meta_path, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, ensure_ascii=False, indent=2)
        logger.info(f"✅ Metadata saved: {meta_path}")

        # ✅ Attention 데이터 저장 (있는 경우)
        attention_data = prediction_results.get("attention_data")
        if attention_data:
            attention_save_data = {
                "image_base64": attention_data.get("image", ""),
                "feature_importance": attention_data.get("feature_importance", {}),
                "temporal_importance": attention_data.get("temporal_importance", {})
            }
            
            with open(attention_path, "w", encoding="utf-8") as fp:
                json.dump(attention_save_data, fp, ensure_ascii=False, indent=2)
            logger.info(f"✅ Attention saved: {attention_path}")

        # ✅ 이동평균 데이터 저장 (있는 경우)
        ma_results = prediction_results.get("ma_results")
        ma_file = None
        if ma_results:
            ma_path = file_predictions_dir / f"prediction_start_{first_date:%Y%m%d}_ma.json"
            try:
                with open(ma_path, "w", encoding="utf-8") as fp:
                    json.dump(ma_results, fp, ensure_ascii=False, indent=2, default=str)
                logger.info(f"✅ MA results saved: {ma_path}")
                ma_file = str(ma_path)
            except Exception as e:
                logger.warning(f"⚠️ Failed to save MA results: {str(e)}")

        # ✅ 인덱스 업데이트
        update_predictions_index_simple(meta)
        
        logger.info(f"✅ Complete prediction save → start date: {meta['prediction_start_date']}")
        return {
            "success": True, 
            "csv_file": str(csv_path),
            "meta_file": str(meta_path),
            "attention_file": str(attention_path) if attention_data else None,
            "ma_file": ma_file,
            "prediction_start_date": meta["prediction_start_date"]
        }

    except Exception as e:
        logger.error(f"❌ save_prediction_simple 오류: {e}")
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

# 2. Attention 데이터를 포함한 로드 함수
def load_prediction_simple(prediction_start_date):
    """
    단순화된 예측 결과 로드 함수
    """
    try:
        predictions_dir = Path(PREDICTIONS_DIR)
        
        if isinstance(prediction_start_date, str):
            start_date = pd.to_datetime(prediction_start_date)
        else:
            start_date = prediction_start_date
        
        date_str = start_date.strftime('%Y%m%d')
        
        csv_filepath = predictions_dir / f"prediction_{date_str}.csv"
        meta_filepath = predictions_dir / f"prediction_{date_str}_meta.json"
        
        if not csv_filepath.exists() or not meta_filepath.exists():
            return {'success': False, 'error': f'Prediction files not found for {start_date.strftime("%Y-%m-%d")}'}
        
        # CSV 로드
        predictions_df = pd.read_csv(csv_filepath)
        predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        if 'Prediction_From' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['Prediction_From'])
        
        predictions = predictions_df.to_dict('records')
        
        # 메타데이터 로드
        with open(meta_filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        logger.info(f"Simple prediction load completed: {len(predictions)} predictions")
        
        return {
            'success': True,
            'predictions': predictions,
            'metadata': metadata,
            'attention_data': {
                'image': metadata.get('attention_map', {}).get('image_base64', ''),
                'feature_importance': metadata.get('attention_map', {}).get('feature_importance', {}),
                'temporal_importance': metadata.get('attention_map', {}).get('temporal_importance', {})
            }
        }
        
    except Exception as e:
        logger.error(f"Error loading prediction: {str(e)}")
        return {'success': False, 'error': str(e)}

def update_predictions_index_simple(metadata):
    """단순화된 예측 인덱스 업데이트 - 파일별 캐시 디렉토리 사용"""
    try:
        # 🔧 metadata가 None인 경우 처리
        if metadata is None:
            logger.warning("⚠️ [INDEX] metadata가 None입니다. 인덱스 업데이트를 건너뜁니다.")
            return False
            
        # 🎯 파일별 캐시 디렉토리 사용
        cache_dirs = get_file_cache_dirs()
        predictions_index_file = cache_dirs['predictions'] / 'predictions_index.csv'
        
        # 기존 인덱스 읽기
        index_data = []
        if predictions_index_file.exists():
            with open(predictions_index_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                index_data = list(reader)
        
        # 중복 제거
        prediction_start_date = metadata.get('prediction_start_date')
        if not prediction_start_date:
            logger.warning("⚠️ [INDEX] metadata에 prediction_start_date가 없습니다.")
            return False
            
        index_data = [row for row in index_data 
                     if row.get('prediction_start_date') != prediction_start_date]
        
        # metrics가 None일 수도 있으므로 안전하게 처리
        metrics = metadata.get('metrics') or {}
        
        # 새 데이터 추가 (🔧 필드명 수정)
        new_row = {
            'prediction_start_date': metadata.get('prediction_start_date', ''),
            'data_end_date': metadata.get('data_end_date', ''),
            'created_at': metadata.get('created_at', ''),
            'semimonthly_period': metadata.get('semimonthly_period', ''),
            'next_semimonthly_period': metadata.get('next_semimonthly_period', ''),
            'prediction_count': metadata.get('total_predictions', metadata.get('prediction_count', 0)),  # 🔧 수정
            'f1_score': metrics.get('f1', 0) if isinstance(metrics, dict) else 0,
            'accuracy': metrics.get('accuracy', 0) if isinstance(metrics, dict) else 0,
            'mape': metrics.get('mape', 0) if isinstance(metrics, dict) else 0,
            'weighted_score': metrics.get('weighted_score', 0) if isinstance(metrics, dict) else 0
        }
        index_data.append(new_row)
        
        # 날짜순 정렬 후 저장
        index_data.sort(key=lambda x: x.get('data_end_date', ''), reverse=True)
        
        if index_data:
            fieldnames = new_row.keys()  # 🔧 일관된 필드명 사용
            with open(predictions_index_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(index_data)
            
            logger.info(f"✅ Predictions index updated successfully: {len(index_data)} entries")
            logger.info(f"📄 Index file: {predictions_index_file}")
            return True
        else:
            logger.warning("⚠️ No data to write to index file")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error updating simple predictions index: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def rebuild_predictions_index_from_existing_files():
    """
    기존 예측 파일들로부터 predictions_index.csv를 재생성하는 함수
    🔧 누적 예측이 기존 단일 예측 캐시를 인식할 수 있도록 함
    """
    try:
        current_file = prediction_state.get('current_file')
        if not current_file:
            logger.warning("⚠️ No current file set, cannot rebuild index")
            return False
        
        cache_dirs = get_file_cache_dirs(current_file)
        predictions_dir = cache_dirs['predictions']
        predictions_index_file = predictions_dir / 'predictions_index.csv'
        
        logger.info(f"🔄 Rebuilding predictions index from existing files in: {predictions_dir}")
        
        # 기존 메타 파일들 찾기
        meta_files = list(predictions_dir.glob("*_meta.json"))
        logger.info(f"📋 Found {len(meta_files)} meta files")
        
        if not meta_files:
            logger.warning("⚠️ No meta files found to rebuild index")
            return False
        
        index_data = []
        
        for meta_file in meta_files:
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 인덱스 레코드 생성 (동일한 필드명 사용)
                new_row = {
                    'prediction_start_date': metadata.get('prediction_start_date', ''),
                    'data_end_date': metadata.get('data_end_date', ''),
                    'created_at': metadata.get('created_at', ''),
                    'semimonthly_period': metadata.get('semimonthly_period', ''),
                    'next_semimonthly_period': metadata.get('next_semimonthly_period', ''),
                    'prediction_count': metadata.get('total_predictions', metadata.get('prediction_count', 0)),
                    'f1_score': metadata.get('metrics', {}).get('f1', 0),
                    'accuracy': metadata.get('metrics', {}).get('accuracy', 0),
                    'mape': metadata.get('metrics', {}).get('mape', 0),
                    'weighted_score': metadata.get('metrics', {}).get('weighted_score', 0)
                }
                
                index_data.append(new_row)
                logger.info(f"  ✅ {meta_file.name}: {new_row['prediction_start_date']}")
                
            except Exception as e:
                logger.warning(f"  ⚠️  Error reading {meta_file.name}: {str(e)}")
                continue
        
        if not index_data:
            logger.error("❌ No valid metadata found")
            return False
        
        # 날짜순 정렬
        index_data.sort(key=lambda x: x.get('data_end_date', ''), reverse=True)
        
        # CSV 파일 생성
        fieldnames = index_data[0].keys()
        
        with open(predictions_index_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(index_data)
        
        logger.info(f"✅ Successfully rebuilt predictions_index.csv with {len(index_data)} entries")
        logger.info(f"📄 Index file: {predictions_index_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error rebuilding predictions index: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def load_prediction_from_csv(prediction_start_date_or_data_end_date):
    """
    하위 호환성을 위한 함수 - 자동으로 새로운 함수로 리다이렉트
    """
    logger.info("Using compatibility wrapper - redirecting to new smart cache function")
    return load_prediction_with_attention_from_csv(prediction_start_date_or_data_end_date)

def load_prediction_with_attention_from_csv_in_dir(prediction_start_date, file_predictions_dir):
    """
    파일별 디렉토리에서 저장된 예측 결과와 attention 데이터를 함께 불러오는 함수
    """
    try:
        if isinstance(prediction_start_date, str):
            start_date = pd.to_datetime(prediction_start_date)
        else:
            start_date = prediction_start_date
        
        date_str = start_date.strftime('%Y%m%d')
        
        # 파일별 디렉토리에서 파일 경로 설정
        csv_filepath = file_predictions_dir / f"prediction_start_{date_str}.csv"
        meta_filepath = file_predictions_dir / f"prediction_start_{date_str}_meta.json"
        attention_filepath = file_predictions_dir / f"prediction_start_{date_str}_attention.json"
        ma_filepath = file_predictions_dir / f"prediction_start_{date_str}_ma.json"
        
        logger.info(f"📂 Loading from file directory: {file_predictions_dir.name}")
        logger.info(f"  📄 CSV: {csv_filepath.name}")
        
        if not csv_filepath.exists() or not meta_filepath.exists():
            logger.warning(f"  ❌ Required files missing in {file_predictions_dir.name}")
            return {'success': False, 'error': f'Prediction files not found for {start_date.strftime("%Y-%m-%d")}'}
        
        # CSV 로드
        predictions_df = pd.read_csv(csv_filepath)
        
        # 🔧 컬럼명 호환성 처리: 소문자로 저장된 컬럼을 대문자로 변환
        if 'date' in predictions_df.columns:
            predictions_df['Date'] = pd.to_datetime(predictions_df['date'])
        elif 'Date' in predictions_df.columns:
            predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        
        if 'prediction' in predictions_df.columns:
            predictions_df['Prediction'] = predictions_df['prediction']
        
        if 'prediction_from' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['prediction_from'])
        elif 'Prediction_From' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['Prediction_From'])
        
        predictions = predictions_df.to_dict('records')
        
        # ✅ 캐시에서 로드할 때 실제값 다시 설정 (현재 파일 데이터 사용)
        try:
            current_file = prediction_state.get('current_file')
            if current_file:
                df = load_data(current_file)
                if df is not None and not df.empty:
                    last_data_date = df.index.max()
                    updated_count = 0
                    
                    # 각 예측에 대해 실제값 확인 및 설정
                    for pred in predictions:
                        pred_date = pd.to_datetime(pred['Date'])
                        
                        # 실제 데이터가 존재하는 날짜면 실제값 설정
                        if (pred_date in df.index and 
                            pd.notna(df.loc[pred_date, 'MOPJ']) and 
                            pred_date <= last_data_date):
                            actual_val = float(df.loc[pred_date, 'MOPJ'])
                            pred['Actual'] = actual_val
                            updated_count += 1
                            logger.debug(f"  📊 Set actual value for {pred_date.strftime('%Y-%m-%d')}: {actual_val:.2f}")
                        elif 'Actual' not in pred or pred['Actual'] is None:
                            pred['Actual'] = None
                    
                    if updated_count > 0:
                        logger.info(f"  🔄 Updated {updated_count} actual values from current data file")
                else:
                    logger.warning(f"  ⚠️  Could not load current data file for actual values")
            else:
                logger.warning(f"  ⚠️  No current file set for actual value update")
        except Exception as e:
            logger.warning(f"  ⚠️  Error updating actual values: {str(e)}")
        
        # 메타데이터 로드
        with open(meta_filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Attention 데이터 로드
        attention_data = {}
        if attention_filepath.exists():
            try:
                with open(attention_filepath, 'r', encoding='utf-8') as f:
                    attention_raw = json.load(f)
                attention_data = {
                    'image': attention_raw.get('image_base64', ''),
                    'feature_importance': attention_raw.get('feature_importance', {}),
                    'temporal_importance': attention_raw.get('temporal_importance', {})
                }
                logger.info(f"  🧠 Attention data loaded successfully")
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to load attention data: {str(e)}")
        
        # 이동평균 데이터 로드
        ma_results = {}
        if ma_filepath.exists():
            try:
                with open(ma_filepath, 'r', encoding='utf-8') as f:
                    ma_results = json.load(f)
                logger.info(f"  📊 MA results loaded successfully")
            except Exception as e:
                logger.warning(f"  ⚠️  Failed to load MA results: {str(e)}")
        
        logger.info(f"✅ File directory cache load completed: {len(predictions)} predictions")
        
        return {
            'success': True,
            'predictions': predictions,
            'metadata': metadata,
            'attention_data': attention_data,
            'ma_results': ma_results
        }
        
    except Exception as e:
        logger.error(f"❌ Error loading prediction from file directory: {str(e)}")
        return {'success': False, 'error': str(e)}

def load_prediction_with_attention_from_csv(prediction_start_date):
    """
    저장된 예측 결과와 attention 데이터를 함께 불러오는 함수 - 파일별 캐시 시스템 사용
    """
    try:
        # 🎯 파일별 캐시 디렉토리 사용
        current_file = prediction_state.get('current_file')
        if not current_file:
            logger.error("❌ No current file set in prediction_state")
            return {'success': False, 'error': 'No current file context available'}
            
        cache_dirs = get_file_cache_dirs(current_file)
        predictions_dir = cache_dirs['predictions']
        
        if isinstance(prediction_start_date, str):
            start_date = pd.to_datetime(prediction_start_date)
        else:
            start_date = prediction_start_date
        
        date_str = start_date.strftime('%Y%m%d')
        
        # 파일 경로들
        csv_filepath = predictions_dir / f"prediction_start_{date_str}.csv"
        meta_filepath = predictions_dir / f"prediction_start_{date_str}_meta.json"
        attention_filepath = predictions_dir / f"prediction_start_{date_str}_attention.json"
        
        # 필수 파일 존재 확인
        if not csv_filepath.exists() or not meta_filepath.exists():
            return {
                'success': False,
                'error': f'Prediction files not found for start date {start_date.strftime("%Y-%m-%d")}'
            }
        
        # CSV 파일 읽기
        predictions_df = pd.read_csv(csv_filepath)
        
        # 🔧 컬럼명 호환성 처리: 소문자로 저장된 컬럼을 대문자로 변환
        if 'date' in predictions_df.columns:
            predictions_df['Date'] = pd.to_datetime(predictions_df['date'])
        elif 'Date' in predictions_df.columns:
            predictions_df['Date'] = pd.to_datetime(predictions_df['Date'])
        
        if 'prediction' in predictions_df.columns:
            predictions_df['Prediction'] = predictions_df['prediction']
        
        if 'prediction_from' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['prediction_from'])
        elif 'Prediction_From' in predictions_df.columns:
            predictions_df['Prediction_From'] = pd.to_datetime(predictions_df['Prediction_From'])
        
        predictions = predictions_df.to_dict('records')
        
        # ✅ 캐시에서 로드할 때 실제값 다시 설정 (현재 파일 데이터 사용)
        try:
            if current_file:
                df = load_data(current_file)
                if df is not None and not df.empty:
                    last_data_date = df.index.max()
                    updated_count = 0
                    
                    # 각 예측에 대해 실제값 확인 및 설정
                    for pred in predictions:
                        pred_date = pd.to_datetime(pred['Date'])
                        
                        # 실제 데이터가 존재하는 날짜면 실제값 설정
                        if (pred_date in df.index and 
                            pd.notna(df.loc[pred_date, 'MOPJ']) and 
                            pred_date <= last_data_date):
                            actual_val = float(df.loc[pred_date, 'MOPJ'])
                            pred['Actual'] = actual_val
                            updated_count += 1
                            logger.debug(f"  📊 Set actual value for {pred_date.strftime('%Y-%m-%d')}: {actual_val:.2f}")
                        elif 'Actual' not in pred or pred['Actual'] is None:
                            pred['Actual'] = None
                    
                    if updated_count > 0:
                        logger.info(f"  🔄 Updated {updated_count} actual values from current data file")
                else:
                    logger.warning(f"  ⚠️  Could not load current data file for actual values")
            else:
                logger.warning(f"  ⚠️  No current file set for actual value update")
        except Exception as e:
            logger.warning(f"  ⚠️  Error updating actual values: {str(e)}")
        
        # 메타데이터 읽기
        with open(meta_filepath, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Attention 데이터 읽기 (있는 경우)
        attention_data = None
        if attention_filepath.exists():
            try:
                with open(attention_filepath, 'r', encoding='utf-8') as f:
                    stored_attention = json.load(f)
                
                attention_data = {
                    'image': stored_attention.get('image_base64', ''),
                    'file_path': None,  # 이미지는 base64로 저장됨
                    'feature_importance': stored_attention.get('feature_importance', {}),
                    'temporal_importance': stored_attention.get('temporal_importance', {})
                }
                logger.info(f"Attention data loaded from: {attention_filepath}")
            except Exception as e:
                logger.warning(f"Failed to load attention data: {str(e)}")
                attention_data = None

        # 🔄 이동평균 데이터 읽기 (있는 경우)
        ma_filepath = predictions_dir / f"prediction_start_{date_str}_ma.json"
        ma_results = None
        if ma_filepath.exists():
            try:
                with open(ma_filepath, 'r', encoding='utf-8') as f:
                    ma_results = json.load(f)
                logger.info(f"MA results loaded from: {ma_filepath} ({len(ma_results)} windows)")
            except Exception as e:
                logger.warning(f"Failed to load MA results: {str(e)}")
                ma_results = None
        
        logger.info(f"Complete prediction data loaded: {csv_filepath} ({len(predictions)} predictions)")
        
        return {
            'success': True,
            'predictions': predictions,
            'metadata': metadata,
            'attention_data': attention_data,
            'ma_results': ma_results,  # 🔑 이동평균 데이터 추가
            'prediction_start_date': start_date.strftime('%Y-%m-%d'),
            'data_end_date': metadata.get('data_end_date'),
            'semimonthly_period': metadata['semimonthly_period'],
            'next_semimonthly_period': metadata['next_semimonthly_period'],
            'metrics': metadata['metrics'],
            'interval_scores': metadata['interval_scores'],
            'selected_features': metadata['selected_features'],
            'has_cached_attention': attention_data is not None,
            'has_cached_ma': ma_results is not None  # 🔑 MA 캐시 여부 추가
        }
        
    except Exception as e:
        logger.error(f"Error loading prediction with attention: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'error': str(e)
        }

def get_saved_predictions_list_for_file(file_path, limit=100):
    """
    특정 파일의 캐시 디렉토리에서 저장된 예측 결과 목록을 조회하는 함수
    
    Parameters:
    -----------
    file_path : str
        현재 파일 경로
    limit : int
        반환할 최대 개수
    
    Returns:
    --------
    list : 저장된 예측 목록
    """
    try:
        predictions_list = []
        
        # 파일별 캐시 디렉토리 경로 구성
        cache_dirs = get_file_cache_dirs(file_path)
        predictions_dir = Path(cache_dirs['predictions'])
        predictions_index_file = predictions_dir / 'predictions_index.csv'
        
        logger.info(f"🔍 [CACHE] Searching predictions in: {predictions_dir}")
        
        if predictions_index_file.exists():
            with open(predictions_index_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if len(predictions_list) >= limit:
                        break
                    
                    prediction_start_date = row.get('prediction_start_date', row.get('first_prediction_date'))
                    data_end_date = row.get('data_end_date', row.get('prediction_base_date', row.get('prediction_date')))
                    
                    if prediction_start_date and data_end_date:
                        pred_info = {
                            'prediction_start_date': prediction_start_date,
                            'data_end_date': data_end_date,
                            'prediction_date': data_end_date,
                            'first_prediction_date': prediction_start_date,
                            'created_at': row.get('created_at'),
                            'semimonthly_period': row.get('semimonthly_period'),
                            'next_semimonthly_period': row.get('next_semimonthly_period'),
                            'prediction_count': row.get('prediction_count'),
                            'actual_business_days': row.get('actual_business_days'),
                            'csv_file': row.get('csv_file'),
                            'meta_file': row.get('meta_file'),
                            'f1_score': float(row.get('f1_score', 0)),
                            'accuracy': float(row.get('accuracy', 0)),
                            'mape': float(row.get('mape', 0)),
                            'weighted_score': float(row.get('weighted_score', 0)),
                            'naming_scheme': row.get('naming_scheme', 'file_based'),
                            'source_file': os.path.basename(file_path),
                            'cache_system': 'file_based'
                        }
                        predictions_list.append(pred_info)
            
            logger.info(f"🎯 [CACHE] Found {len(predictions_list)} predictions in file-specific cache")
        else:
            logger.info(f"📂 [CACHE] No predictions index found in {predictions_index_file}")
        
        # 날짜순으로 정렬 (최신 순)
        predictions_list.sort(key=lambda x: x['data_end_date'], reverse=True)
        
        return predictions_list
        
    except Exception as e:
        logger.error(f"Error reading file-specific predictions list: {str(e)}")
        return []

def get_saved_predictions_list(limit=100):
    """
    저장된 예측 결과 목록을 조회하는 함수 (새로운 파일 체계 호환)
    
    Parameters:
    -----------
    limit : int
        반환할 최대 개수
    
    Returns:
    --------
    list : 저장된 예측 목록
    """
    try:
        predictions_list = []
        
        # 1. 파일별 캐시 시스템에서 예측 검색
        cache_root = Path(CACHE_ROOT_DIR)
        if cache_root.exists():
            for file_dir in cache_root.iterdir():
                if not file_dir.is_dir():
                    continue
                
                predictions_dir = file_dir / 'predictions'
                predictions_index_file = predictions_dir / 'predictions_index.csv'
                
                if predictions_index_file.exists():
                    with open(predictions_index_file, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if len(predictions_list) >= limit:
                                break
                            
                            prediction_start_date = row.get('prediction_start_date', row.get('first_prediction_date'))
                            data_end_date = row.get('data_end_date', row.get('prediction_base_date', row.get('prediction_date')))
                            
                            if prediction_start_date and data_end_date:
                                pred_info = {
                                    'prediction_start_date': prediction_start_date,
                                    'data_end_date': data_end_date,
                                    'prediction_date': data_end_date,
                                    'first_prediction_date': prediction_start_date,
                                    'created_at': row.get('created_at'),
                                    'semimonthly_period': row.get('semimonthly_period'),
                                    'next_semimonthly_period': row.get('next_semimonthly_period'),
                                    'prediction_count': row.get('prediction_count'),
                                    'actual_business_days': row.get('actual_business_days'),
                                    'csv_file': row.get('csv_file'),
                                    'meta_file': row.get('meta_file'),
                                    'f1_score': float(row.get('f1_score', 0)),
                                    'accuracy': float(row.get('accuracy', 0)),
                                    'mape': float(row.get('mape', 0)),
                                    'weighted_score': float(row.get('weighted_score', 0)),
                                    'naming_scheme': row.get('naming_scheme', 'file_based'),
                                    'source_file': file_dir.name,
                                    'cache_system': 'file_based'
                                }
                                predictions_list.append(pred_info)
        
        if len(predictions_list) == 0:
            logger.info("No predictions found in file-based cache system")
        
        # 날짜순으로 정렬 (최신 순)
        predictions_list.sort(key=lambda x: x['data_end_date'], reverse=True)
        
        logger.info(f"Retrieved {len(predictions_list)} predictions from cache systems")
        return predictions_list
        
    except Exception as e:
        logger.error(f"Error reading predictions list: {str(e)}")
        return []

def load_accumulated_predictions_from_csv(start_date, end_date=None, limit=None, file_path=None):
    """
    CSV에서 누적 예측 결과를 빠르게 불러오는 함수
    새로운 파일명 체계와 스마트 캐시 시스템 사용
    
    Parameters:
    -----------
    start_date : str or datetime
        시작 날짜 (데이터 기준일)
    end_date : str or datetime, optional
        종료 날짜 (데이터 기준일)
    limit : int, optional
        최대 로드할 예측 개수
    file_path : str, optional
        현재 파일 경로 (해당 파일의 캐시 디렉토리에서만 검색)
    
    Returns:
    --------
    list : 누적 예측 결과 리스트
    """
    try:
        logger.info(f"🔍 [CACHE_LOAD] Starting accumulated predictions load")
        logger.info(f"🔍 [CACHE_LOAD] Input params: start_date={start_date}, end_date={end_date}, file_path={file_path}")
        
        # 날짜 형식 통일
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date and isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        logger.info(f"🔍 [CACHE_LOAD] Loading accumulated predictions from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d') if end_date else 'latest'}")
        
        # 저장된 예측 목록 조회 (파일별 캐시 디렉토리 사용)
        all_predictions = []
        if file_path:
            logger.info(f"🔍 [CACHE_LOAD] Searching in file-specific cache directory for {os.path.basename(file_path)}")
            try:
                all_predictions = get_saved_predictions_list_for_file(file_path, limit=1000)  # ✅ 파일별 검색
                logger.info(f"🎯 [CACHE_LOAD] Found {len(all_predictions)} prediction files in cache")
            except Exception as e:
                logger.error(f"❌ [CACHE_LOAD] Error in get_saved_predictions_list_for_file: {str(e)}")
                logger.error(traceback.format_exc())
                return []
        else:
            logger.info(f"🔍 [CACHE_LOAD] Searching in global cache directory (legacy mode)")
            try:
                all_predictions = get_saved_predictions_list(limit=1000)  # 전체 검색 (하위 호환)
                logger.info(f"🎯 [CACHE_LOAD] Found {len(all_predictions)} prediction files in legacy cache")
            except Exception as e:
                logger.error(f"❌ [CACHE_LOAD] Error in get_saved_predictions_list: {str(e)}")
                logger.error(traceback.format_exc())
                return []
        
        # 날짜 범위 필터링 (데이터 기준일 기준)
        filtered_predictions = []
        for pred_info in all_predictions:
            # 인덱스에서 데이터 기준일 확인
            data_end_date = pd.to_datetime(pred_info.get('data_end_date', pred_info.get('prediction_date')))
            
            # 날짜 범위 확인
            if data_end_date >= start_date:
                if end_date is None or data_end_date <= end_date:
                    filtered_predictions.append(pred_info)
            
            # 제한 개수 확인
            if limit and len(filtered_predictions) >= limit:
                break
        
        logger.info(f"📋 [CACHE] Found {len(filtered_predictions)} matching prediction files in date range")
        if len(filtered_predictions) > 0:
            logger.info(f"📅 [CACHE] Available cached dates:")
            for pred in filtered_predictions:
                data_end_date = pred.get('data_end_date', pred.get('prediction_date'))
                logger.info(f"    - {data_end_date}")
        
        # 각 예측 결과 로드
        accumulated_results = []
        for i, pred_info in enumerate(filtered_predictions):
            try:
                # 데이터 기준일을 사용하여 예측 시작일 계산
                data_end_date = pd.to_datetime(pred_info.get('data_end_date', pred_info.get('prediction_date')))
                
                # 데이터 기준일로부터 예측 시작일 계산
                prediction_start_date = data_end_date + pd.Timedelta(days=1)
                while prediction_start_date.weekday() >= 5 or is_holiday(prediction_start_date):
                    prediction_start_date += pd.Timedelta(days=1)
                
                # 파일별 캐시 디렉토리 사용
                if file_path:
                    cache_dirs = get_file_cache_dirs(file_path)
                    loaded_result = load_prediction_with_attention_from_csv_in_dir(prediction_start_date, cache_dirs['predictions'])
                else:
                    loaded_result = load_prediction_with_attention_from_csv(prediction_start_date)
                
                if loaded_result['success']:
                    logger.info(f"  ✅ [CACHE] Successfully loaded cached prediction for {data_end_date.strftime('%Y-%m-%d')}")
                    # 누적 예측 형식에 맞게 변환
                    # 안전한 데이터 구조 생성
                    predictions = loaded_result.get('predictions', [])
                    
                    # 예측 데이터가 중첩된 딕셔너리 구조인 경우 처리
                    if isinstance(predictions, dict):
                        if 'future' in predictions:
                            predictions = predictions['future']
                        elif 'predictions' in predictions:
                            predictions = predictions['predictions']
                    
                    if not isinstance(predictions, list):
                        logger.warning(f"Loaded predictions is not a list for {data_end_date.strftime('%Y-%m-%d')}: {type(predictions)}")
                        predictions = []
                    
                    metadata = loaded_result.get('metadata', {})
                    if not isinstance(metadata, dict):
                        metadata = {}
                    
                    # 🔧 metrics 안전성 처리: None이면 기본값 설정
                    cached_metrics = metadata.get('metrics')
                    if not cached_metrics or not isinstance(cached_metrics, dict):
                        cached_metrics = {'f1': 0.0, 'accuracy': 0.0, 'mape': 0.0, 'weighted_score': 0.0}
                    
                    accumulated_item = {
                        'date': data_end_date.strftime('%Y-%m-%d'),  # 데이터 기준일
                        'prediction_start_date': loaded_result.get('prediction_start_date'),  # 예측 시작일
                        'predictions': predictions,
                        'metrics': cached_metrics,
                        'interval_scores': metadata.get('interval_scores', {}),
                        'next_semimonthly_period': metadata.get('next_semimonthly_period'),
                        'actual_business_days': metadata.get('actual_business_days'),
                        'original_interval_scores': metadata.get('interval_scores', {}),
                        'has_attention': loaded_result.get('has_cached_attention', False)
                    }
                    accumulated_results.append(accumulated_item)
                    logger.info(f"  ✅ [CACHE] Added to results {i+1}/{len(filtered_predictions)}: {data_end_date.strftime('%Y-%m-%d')}")
                else:
                    logger.warning(f"  ❌ [CACHE] Failed to load prediction {i+1}/{len(filtered_predictions)}: {loaded_result.get('error')}")
                    
            except Exception as e:
                logger.error(f"  ❌ Error loading prediction {i+1}/{len(filtered_predictions)}: {str(e)}")
                continue
        
        logger.info(f"🎯 [CACHE] Successfully loaded {len(accumulated_results)} predictions from CSV cache files")
        return accumulated_results
        
    except Exception as e:
        logger.error(f"Error loading accumulated predictions from CSV: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def delete_saved_prediction(prediction_date):
    """
    저장된 예측 결과를 삭제하는 함수
    
    Parameters:
    -----------
    prediction_date : str or datetime
        삭제할 예측 날짜
    
    Returns:
    --------
    dict : 삭제 결과
    """
    try:
        # 날짜 형식 통일
        if isinstance(prediction_date, str):
            pred_date = pd.to_datetime(prediction_date)
        else:
            pred_date = prediction_date
        
        date_str = pred_date.strftime('%Y%m%d')
        
        # 파일 경로들 (TARGET_DATE 방식)
        csv_filepath = os.path.join(PREDICTIONS_DIR, f"prediction_target_{date_str}.csv")
        meta_filepath = os.path.join(PREDICTIONS_DIR, f"prediction_target_{date_str}_meta.json")
        
        # 파일 삭제
        deleted_files = []
        if os.path.exists(csv_filepath):
            os.remove(csv_filepath)
            deleted_files.append(csv_filepath)
        
        if os.path.exists(meta_filepath):
            os.remove(meta_filepath)
            deleted_files.append(meta_filepath)
        
        # 🚫 레거시 인덱스 제거 기능은 파일별 캐시 시스템에서 제거됨
        # 파일별 캐시에서는 각 파일의 predictions_index.csv가 자동으로 관리됨
        logger.info("⚠️ Legacy delete_saved_prediction function called - not supported in file-based cache system")
        
        return {
            'success': True,
            'deleted_files': deleted_files,
            'message': f'Prediction for {pred_date.strftime("%Y-%m-%d")} deleted successfully'
        }
        
    except Exception as e:
        logger.error(f"Error deleting saved prediction: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

#######################################################################
# 예측 신뢰도 및 구매 신뢰도 계산 함수
#######################################################################

def calculate_prediction_consistency(accumulated_predictions, target_period):
    """
    다음 반월에 대한 여러 날짜의 예측 일관성을 계산
    
    Parameters:
    -----------
    accumulated_predictions: list
        여러 날짜에 수행한 예측 결과 목록
    target_period: str
        다음 반월 기간 (예: "2025-01-SM1")
    
    Returns:
    -----------
    dict: 일관성 점수와 관련 메트릭
    """
    import numpy as np
    
    # 날짜별 예측 데이터 추출
    period_predictions = {}
    
    for prediction in accumulated_predictions:
        # 안전한 데이터 접근
        if not isinstance(prediction, dict):
            continue
            
        prediction_date = prediction.get('date')
        next_period = prediction.get('next_semimonthly_period')
        predictions_list = prediction.get('predictions', [])
        
        if next_period != target_period:
            continue
            
        if prediction_date not in period_predictions:
            period_predictions[prediction_date] = []
        
        # predictions_list가 배열인지 확인
        if not isinstance(predictions_list, list):
            logger.warning(f"predictions_list is not a list for {prediction_date}: {type(predictions_list)}")
            continue
            
        for pred in predictions_list:
            # pred가 딕셔너리인지 확인
            if not isinstance(pred, dict):
                logger.warning(f"Prediction item is not a dict for {prediction_date}: {type(pred)}")
                continue
                
            pred_date = pred.get('Date') or pred.get('date')
            pred_value = pred.get('Prediction') or pred.get('prediction')
            
            # 값이 유효한지 확인
            if pred_date and pred_value is not None:
                period_predictions[prediction_date].append({
                    'date': pred_date,
                    'value': pred_value
                })
    
    # 날짜별로 정렬
    prediction_dates = sorted(period_predictions.keys())
    
    if len(prediction_dates) < 2:
        return {
            "consistency_score": None,
            "message": "Insufficient prediction data (min 2 required)",
            "period": target_period,
            "dates_count": len(prediction_dates)
        }
    
    # 일관성 분석을 위한 날짜 매핑
    date_predictions = {}
    
    for pred_date in prediction_dates:
        for p in period_predictions[pred_date]:
            target_date = p['date']
            if target_date not in date_predictions:
                date_predictions[target_date] = []
            
            date_predictions[target_date].append({
                'prediction_date': pred_date,
                'value': p['value']
            })
    
    # 각 타겟 날짜별 예측값 변동성 계산
    overall_variations = []
    
    for target_date, predictions in date_predictions.items():
        if len(predictions) >= 2:
            # 예측값 추출 (None 값 필터링)
            values = [p['value'] for p in predictions if p['value'] is not None]
            
            if len(values) < 2:
                continue
                
            # 값이 모두 같은 경우 CV를 0으로 처리
            if all(v == values[0] for v in values):
                cv = 0.0
                overall_variations.append(cv)
                continue
            
            mean_value = np.mean(values)
            std_value = np.std(values)
            
            # 변동 계수 (Coefficient of Variation)
            cv = std_value / abs(mean_value) if mean_value != 0 else float('inf')
            overall_variations.append(cv)
    
    # 전체 일관성 점수 계산 (변동 계수 평균을 0-100 점수로 변환)
    if overall_variations:
        avg_cv = np.mean(overall_variations)
        consistency_score = max(0, min(100, 100 - (avg_cv * 100)))
    else:
        consistency_score = None
    
    # 신뢰도 등급 부여
    if consistency_score is not None:
        if consistency_score >= 90:
            grade = "Very High"
        elif consistency_score >= 75:
            grade = "High"
        elif consistency_score >= 60:
            grade = "Medium"
        elif consistency_score >= 40:
            grade = "Low"
        else:
            grade = "Very Low"
    else:
        grade = "Unable to determine"
    
    return {
        "consistency_score": consistency_score,
        "consistency_grade": grade,
        "target_period": target_period,
        "prediction_count": len(prediction_dates),
        "average_variation": avg_cv * 100 if overall_variations else None,
        "message": f"Consistency for period {target_period} based on {len(prediction_dates)} predictions"
    }

# 누적 예측의 구매 신뢰도 계산 함수 (올바른 버전)
def calculate_accumulated_purchase_reliability(accumulated_predictions):
    """
    누적 예측의 구매 신뢰도 계산
    각 예측에서 얻은 최고 점수의 합 / (예측 횟수 × 3점)
    
    - 각 예측 날짜마다 최대 3점을 받을 수 있음
    - 전체 최대 점수 = 예측 횟수 × 3점
    - 구매 신뢰도 = 총 획득 점수 / 전체 최대 점수 × 100%
    """
    print(f"🔍 [RELIABILITY] Function called with {len(accumulated_predictions) if accumulated_predictions else 0} predictions")
    
    if not accumulated_predictions or not isinstance(accumulated_predictions, list):
        print(f"⚠️ [RELIABILITY] Invalid input: accumulated_predictions is empty or not a list")
        return 0.0
    
    try:
        total_best_score = 0
        prediction_count = len(accumulated_predictions)
        print(f"📊 [RELIABILITY] Processing {prediction_count} predictions...")
        
        for i, pred in enumerate(accumulated_predictions):
            if not isinstance(pred, dict):
                continue
                
            interval_scores = pred.get('interval_scores', {})
            
            if interval_scores and isinstance(interval_scores, dict):
                # 유효한 interval score 찾기
                valid_scores = []
                for score_data in interval_scores.values():
                    if isinstance(score_data, dict) and 'score' in score_data:
                        score_value = score_data.get('score', 0)
                        if isinstance(score_value, (int, float)):
                            valid_scores.append(score_value)
                
                if valid_scores:
                    best_score = max(valid_scores)
                    # 점수가 3점을 초과하면 3점으로 제한 (3점이 만점)
                    capped_score = min(best_score, 3.0)
                    total_best_score += capped_score
                    
                    print(f"📊 [RELIABILITY] Prediction {i+1} ({pred.get('date')}): original_score={best_score:.1f}, capped_score={capped_score:.1f}, valid_scores={len(valid_scores)}")
                    logger.info(f"📊 날짜 {pred.get('date')}: 원본점수={best_score:.1f}, 적용점수={capped_score:.1f}")
        
        # 전체 누적 구매 신뢰도 = 총 획득 점수 / (예측 횟수 × 3점)
        max_possible_total_score = prediction_count * 3
        
        if max_possible_total_score > 0:
            reliability_percentage = (total_best_score / max_possible_total_score) * 100
        else:
            reliability_percentage = 0.0
        
        print(f"🎯 [RELIABILITY] FINAL CALCULATION:")
        print(f"  - 예측 횟수: {prediction_count}개")
        print(f"  - 총 획득 점수: {total_best_score:.1f}점")
        print(f"  - 최대 가능 점수: {max_possible_total_score}점 ({prediction_count} × 3)")
        print(f"  - 구매 신뢰도: {reliability_percentage:.1f}%")
        
        logger.info(f"🎯 올바른 구매 신뢰도 계산:")
        logger.info(f"  - 예측 횟수: {prediction_count}개")
        logger.info(f"  - 총 획득 점수: {total_best_score:.1f}점")
        logger.info(f"  - 최대 가능 점수: {max_possible_total_score}점 ({prediction_count} × 3)")
        logger.info(f"  - 구매 신뢰도: {reliability_percentage:.1f}%")
        
        # ✅ 추가 검증 로깅
        if reliability_percentage == 100.0:
            logger.warning("⚠️ [SIMPLE_RELIABILITY] 구매 신뢰도가 100%입니다. 각 예측별 점수 확인 필요")
        elif reliability_percentage == 0.0:
            logger.warning("⚠️ [SIMPLE_RELIABILITY] 구매 신뢰도가 0%입니다. 점수 데이터 확인 필요")
        
        return reliability_percentage
            
    except Exception as e:
        logger.error(f"Error calculating accumulated purchase reliability: {str(e)}")
        return 0.0

def calculate_accumulated_purchase_reliability_with_debug(accumulated_predictions):
    """
    디버그 정보와 함께 누적 예측의 구매 신뢰도 계산
    """
    if not accumulated_predictions or not isinstance(accumulated_predictions, list):
        return 0.0, {}
    
    debug_info = {
        'prediction_count': len(accumulated_predictions),
        'individual_scores': [],
        'total_best_score': 0,
        'max_possible_total_score': 0
    }
    
    try:
        total_best_score = 0
        prediction_count = len(accumulated_predictions)
        
        for i, pred in enumerate(accumulated_predictions):
            if not isinstance(pred, dict):
                continue
                
            pred_date = pred.get('date')
            interval_scores = pred.get('interval_scores', {})
            
            best_score = 0
            capped_score = 0  # ✅ 초기화 추가
            valid_scores = []  # ✅ valid_scores도 외부에서 초기화
            
            if interval_scores and isinstance(interval_scores, dict):
                # 유효한 interval score 찾기
                for score_data in interval_scores.values():
                    if isinstance(score_data, dict) and 'score' in score_data:
                        score_value = score_data.get('score', 0)
                        if isinstance(score_value, (int, float)):
                            valid_scores.append(score_value)
                
                if valid_scores:
                    best_score = max(valid_scores)
                    # 점수가 3점을 초과하면 3점으로 제한 (3점이 만점)
                    capped_score = min(best_score, 3.0)
                    total_best_score += capped_score
            
            debug_info['individual_scores'].append({
                'date': pred_date,
                'original_best_score': best_score,
                'capped_score': capped_score,
                'max_score_per_prediction': 3,
                'has_valid_scores': len(valid_scores) > 0
            })
        
        # 전체 계산 - 3점이 만점
        max_possible_total_score = prediction_count * 3
        reliability_percentage = (total_best_score / max_possible_total_score) * 100 if max_possible_total_score > 0 else 0.0
        
        debug_info['total_best_score'] = total_best_score
        debug_info['max_possible_total_score'] = max_possible_total_score
        debug_info['reliability_percentage'] = reliability_percentage
        
        logger.info(f"🎯 올바른 누적 구매 신뢰도 계산:")
        logger.info(f"  - 예측 횟수: {prediction_count}회")
        
        # 🔍 개별 점수 디버깅 정보 출력
        for score_info in debug_info['individual_scores']:
            logger.info(f"📊 날짜 {score_info['date']}: 원본점수={score_info['original_best_score']}, 적용점수={score_info['capped_score']}, 유효점수있음={score_info['has_valid_scores']}")
        
        logger.info(f"  - 총 획득 점수: {total_best_score:.1f}점")
        logger.info(f"  - 최대 가능 점수: {max_possible_total_score}점 ({prediction_count} × 3)")
        logger.info(f"  - 구매 신뢰도: {reliability_percentage:.1f}%")
        
        # ✅ 추가 검증 로깅
        if reliability_percentage == 100.0:
            logger.warning("⚠️ [RELIABILITY] 구매 신뢰도가 100%입니다. 계산 검증:")
            logger.warning(f"   - total_best_score: {total_best_score}")
            logger.warning(f"   - max_possible_total_score: {max_possible_total_score}")
            logger.warning(f"   - prediction_count: {prediction_count}")
            for i, score_info in enumerate(debug_info['individual_scores']):
                logger.warning(f"   - 예측 {i+1}: {score_info}")
        elif reliability_percentage == 0.0:
            logger.warning("⚠️ [RELIABILITY] 구매 신뢰도가 0%입니다. 계산 검증:")
            logger.warning(f"   - total_best_score: {total_best_score}")
            logger.warning(f"   - max_possible_total_score: {max_possible_total_score}")
            logger.warning(f"   - prediction_count: {prediction_count}")
        
        return reliability_percentage, debug_info
            
    except Exception as e:
        logger.error(f"Error calculating accumulated purchase reliability: {str(e)}")
        return 0.0, {'error': str(e)}

def calculate_actual_business_days(predictions):
    """
    예측 결과에서 실제 영업일 수를 계산하는 헬퍼 함수
    """
    if not predictions:
        return 0
    
    try:
        actual_days = len([p for p in predictions 
                          if p.get('Date') and not p.get('is_synthetic', False)])
        return actual_days
    except Exception as e:
        logger.error(f"Error calculating actual business days: {str(e)}")
        return 0

def get_previous_semimonthly_period(semimonthly_period):
    """
    주어진 반월 기간의 이전 반월 기간을 계산하는 함수
    
    Parameters:
    -----------
    semimonthly_period : str
        "YYYY-MM-SM1" 또는 "YYYY-MM-SM2" 형식의 반월 기간
    
    Returns:
    --------
    str
        이전 반월 기간
    """
    parts = semimonthly_period.split('-')
    year = int(parts[0])
    month = int(parts[1])
    half = parts[2]
    
    if half == "SM1":
        # 상반월인 경우 이전 월의 하반월로
        if month == 1:
            return f"{year-1}-12-SM2"
        else:
            return f"{year}-{month-1:02d}-SM2"
    else:
        # 하반월인 경우 같은 월의 상반월로
        return f"{year}-{month:02d}-SM1"

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
    
    # 파일 저장 - 파일별 캐시 디렉토리 사용
    try:
        cache_dirs = get_file_cache_dirs()  # 현재 파일의 캐시 디렉토리 가져오기
        attn_dir = cache_dirs['plots']  # plots 디렉토리에 저장
        
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

def plot_prediction_basic(sequence_df, prediction_start_date, start_day_value, 
                         f1, accuracy, mape, weighted_score_pct, 
                         current_date=None,  # 🔑 추가: 데이터 컷오프 날짜
                         save_prefix=None, title_prefix="Basic Prediction Graph",
                         y_min=None, y_max=None, file_path=None):
    """
    기본 예측 그래프 시각화 - 과거/미래 명확 구분
    🔑 current_date 이후는 미래 예측으로만 표시 (데이터 누출 방지)
    """
    
    fig = None
    
    try:
        logger.info(f"Creating prediction graph for prediction starting {format_date(prediction_start_date)}")
        
        # 📁 저장 디렉토리 설정 (파일별 캐시 디렉토리 사용)
        if save_prefix is None:
            try:
                cache_dirs = get_file_cache_dirs(file_path)
                save_dir = cache_dirs['plots']
            except Exception as e:
                logger.warning(f"Could not get cache directories for plots: {str(e)}")
                save_dir = Path("temp_plots")
                save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = Path(save_prefix)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # DataFrame의 날짜 열이 문자열인 경우 날짜 객체로 변환
        if 'Date' in sequence_df.columns and isinstance(sequence_df['Date'].iloc[0], str):
            sequence_df['Date'] = pd.to_datetime(sequence_df['Date'])
        
        # ✅ current_date 기준으로 과거/미래 분할
        if current_date is not None:
            current_date = pd.to_datetime(current_date)
            
            # 과거 데이터 (current_date 이전): 실제값과 예측값 모두 표시 가능
            past_df = sequence_df[sequence_df['Date'] <= current_date].copy()
            # 미래 데이터 (current_date 이후): 예측값만 표시
            future_df = sequence_df[sequence_df['Date'] > current_date].copy()
            
            # 과거 데이터에서 실제값이 있는 것만 검증용으로 사용
            valid_df = past_df.dropna(subset=['Actual']) if 'Actual' in past_df.columns else pd.DataFrame()
            
            logger.info(f"  📊 Data split - Past: {len(past_df)}, Future: {len(future_df)}, Validation: {len(valid_df)}")
        else:
            # current_date가 없으면 기존 방식 사용 (하위 호환성)
            valid_df = sequence_df.dropna(subset=['Actual']) if 'Actual' in sequence_df.columns else pd.DataFrame()
            future_df = sequence_df
            past_df = valid_df
        
        pred_df = sequence_df.dropna(subset=['Prediction'])
        
        # 그래프 생성
        fig = plt.figure(figsize=(16, 9))
        plt.subplots_adjust(top=0.85)
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3, figure=fig)
        
        # 그래프 타이틀과 서브타이틀
        if isinstance(prediction_start_date, str):
            main_title = f"{title_prefix} - Start: {prediction_start_date}"
        else:
            main_title = f"{title_prefix} - Start: {prediction_start_date.strftime('%Y-%m-%d')}"
        
        # ✅ 과거/미래 구분 정보가 포함된 서브타이틀
        if current_date is not None:
            validation_count = len(valid_df)
            future_count = len(future_df)
            subtitle = f"Data Cutoff: {current_date.strftime('%Y-%m-%d')} | Validation: {validation_count} pts | Future: {future_count} pts"
            if validation_count > 0:
                subtitle += f" | F1: {f1:.3f}, Acc: {accuracy:.1f}%, MAPE: {mape:.1f}%"
        else:
            # 기존 방식
            if f1 == 0 and accuracy == 0 and mape == 0 and weighted_score_pct == 0:
                subtitle = "Future Prediction Only (No Validation Data Available)"
            else:
                subtitle = f"F1: {f1:.2f}, Acc: {accuracy:.2f}%, MAPE: {mape:.2f}%, Score: {weighted_score_pct:.2f}%"

        fig.text(0.5, 0.94, main_title, ha='center', fontsize=14, weight='bold')
        fig.text(0.5, 0.90, subtitle, ha='center', fontsize=12)
        
        # (1) 상단: 가격 예측 그래프
        ax1 = fig.add_subplot(gs[0])
        ax1.set_title("Price Prediction: Past Validation vs Future Forecast", fontsize=13)
        ax1.grid(True, linestyle='--', alpha=0.5)

        if y_min is not None and y_max is not None:
            ax1.set_ylim(y_min, y_max)
        
        # 예측 시작 날짜 처리
        if isinstance(prediction_start_date, str):
            start_date = pd.to_datetime(prediction_start_date)
        else:
            start_date = prediction_start_date
        
        # 시작일 이전 날짜 계산 (연결점용)
        prev_date = start_date - pd.Timedelta(days=1)
        while prev_date.weekday() >= 5 or is_holiday(prev_date):
            prev_date -= pd.Timedelta(days=1)
        
        # ✅ 1. 과거 실제값 (파란색 실선) - 가장 중요한 기준선
        if not valid_df.empty:
            real_dates = [prev_date] + valid_df['Date'].tolist()
            real_values = [start_day_value] + valid_df['Actual'].tolist()
            ax1.plot(real_dates, real_values, marker='o', color='blue', 
                    label='Actual (Past)', linewidth=2.5, markersize=5, zorder=3)
        
        # ✅ 2. 과거 예측값 (회색 점선) - 모델 성능 확인용
        if not valid_df.empty:
            past_pred_dates = [prev_date] + valid_df['Date'].tolist()
            past_pred_values = [start_day_value] + valid_df['Prediction'].tolist()
            ax1.plot(past_pred_dates, past_pred_values, marker='x', color='gray', 
                    label='Predicted (Past)', linewidth=1.5, linestyle=':', markersize=4, alpha=0.8, zorder=2)
        
        # ✅ 3. 미래 예측값 (빨간색 점선) - 핵심 예측
        if not future_df.empty:
            future_dates = future_df['Date'].tolist()
            future_values = future_df['Prediction'].tolist()
            
            # 연결선 (마지막 실제값 → 첫 미래 예측값)
            if not valid_df.empty and future_dates:
                # 마지막 검증 데이터의 실제값에서 첫 미래 예측으로 연결
                connection_x = [valid_df['Date'].iloc[-1], future_dates[0]]
                connection_y = [valid_df['Actual'].iloc[-1], future_values[0]]
                ax1.plot(connection_x, connection_y, '--', color='orange', alpha=0.6, linewidth=1.5, zorder=1)
            elif start_day_value is not None and future_dates:
                # 검증 데이터가 없으면 시작값에서 연결
                connection_x = [prev_date, future_dates[0]]
                connection_y = [start_day_value, future_values[0]]
                ax1.plot(connection_x, connection_y, '--', color='orange', alpha=0.6, linewidth=1.5, zorder=1)
            
            ax1.plot(future_dates, future_values, marker='o', color='red', 
                    label='Predicted (Future)', linewidth=2.5, linestyle='--', markersize=5, zorder=3)
        
        # ✅ 4. 데이터 컷오프 라인 (초록색 세로선)
        if current_date is not None:
            ax1.axvline(x=current_date, color='green', linestyle='-', alpha=0.8, 
                       linewidth=2.5, label=f'Data Cutoff', zorder=4)
            
            # 컷오프 날짜 텍스트 추가
            ax1.text(current_date, ax1.get_ylim()[1] * 0.95, 
                    f'{current_date.strftime("%m/%d")}', 
                    ha='center', va='top', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        else:
            # 예측 시작점에 수직선 표시 (기존 방식)
            ax1.axvline(x=start_date, color='green', linestyle='--', alpha=0.7, 
                       linewidth=2, label='Prediction Start', zorder=4)
        
        # ✅ 5. 배경 색칠 (방향성 일치 여부) - 검증 데이터만
        if not valid_df.empty and len(valid_df) > 1:
            for i in range(len(valid_df) - 1):
                curr_date = valid_df['Date'].iloc[i]
                next_date = valid_df['Date'].iloc[i + 1]
                
                curr_actual = valid_df['Actual'].iloc[i]
                next_actual = valid_df['Actual'].iloc[i + 1]
                curr_pred = valid_df['Prediction'].iloc[i]
                next_pred = valid_df['Prediction'].iloc[i + 1]
                
                # 방향 계산
                actual_dir = np.sign(next_actual - curr_actual)
                pred_dir = np.sign(next_pred - curr_pred)
                
                # 방향 일치 여부에 따른 색상
                color = 'lightblue' if actual_dir == pred_dir else 'lightcoral'
                ax1.axvspan(curr_date, next_date, color=color, alpha=0.15, zorder=0)
        
        ax1.set_xlabel("")
        ax1.set_ylabel("Price (USD/MT)", fontsize=11)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # ✅ (2) 하단: 오차 분석 - 검증 데이터만 또는 변화량
        ax2 = fig.add_subplot(gs[1])
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        if not valid_df.empty and len(valid_df) > 0:
            # 검증 데이터의 절대 오차
            error_dates = valid_df['Date'].tolist()
            error_values = [abs(row['Actual'] - row['Prediction']) for _, row in valid_df.iterrows()]
            
            if error_dates and error_values:
                bars = ax2.bar(error_dates, error_values, width=0.6, color='salmon', alpha=0.7, edgecolor='darkred', linewidth=0.5)
                ax2.set_title(f"Prediction Error - Validation Period ({len(error_dates)} points)", fontsize=11)
                
                # 평균 오차 라인
                avg_error = np.mean(error_values)
                ax2.axhline(y=avg_error, color='red', linestyle='--', alpha=0.8, 
                           label=f'Avg Error: {avg_error:.2f}')
                ax2.legend(fontsize=9)
            else:
                ax2.text(0.5, 0.5, "No validation errors to display", 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=10)
                ax2.set_title("Error Analysis")
        else:
            # 실제값이 없는 경우: 미래 예측의 일일 변화량 표시
            if not future_df.empty and len(future_df) > 1:
                change_dates = future_df['Date'].iloc[1:].tolist()
                change_values = np.diff(future_df['Prediction'].values)
                
                # 상승/하락에 따른 색상 구분
                colors = ['green' if change >= 0 else 'red' for change in change_values]
                
                bars = ax2.bar(change_dates, change_values, width=0.6, color=colors, alpha=0.7)
                ax2.set_title("Daily Price Changes - Future Predictions", fontsize=11)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
                
                # 범례 추가
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='green', alpha=0.7, label='Price Up'),
                                 Patch(facecolor='red', alpha=0.7, label='Price Down')]
                ax2.legend(handles=legend_elements, fontsize=9)
            else:
                ax2.text(0.5, 0.5, "Insufficient data for change analysis", 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=10)
                ax2.set_title("Change Analysis")
        
        ax2.set_xlabel("Date", fontsize=11)
        ax2.set_ylabel("Value", fontsize=11)
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 파일 경로 생성
        if isinstance(prediction_start_date, str):
            date_str = pd.to_datetime(prediction_start_date).strftime('%Y%m%d')
        else:
            date_str = prediction_start_date.strftime('%Y%m%d')
        
        filename = f"prediction_start_{date_str}.png"
        full_path = save_dir / filename
        
        # 이미지를 메모리에 저장
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300, bbox_inches='tight')
        img_buf.seek(0)
        
        # Base64로 인코딩
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        
        # 파일로 저장
        plt.savefig(str(full_path), dpi=300, bbox_inches='tight')
        
        # 메모리 정리
        plt.close(fig)
        plt.clf()
        img_buf.close()
        
        logger.info(f"Enhanced prediction graph saved: {full_path}")
        logger.info(f"  - Past validation points: {len(valid_df) if not valid_df.empty else 0}")
        logger.info(f"  - Future prediction points: {len(future_df) if not future_df.empty else 0}")
        
        return str(full_path), img_str
        
    except Exception as e:
        if fig is not None:
            plt.close(fig)
        plt.close('all')
        plt.clf()
        
        logger.error(f"Error in enhanced graph creation: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None
    
def plot_moving_average_analysis(ma_results, sequence_start_date, save_prefix=None,
                               title_prefix="Moving Average Analysis", y_min=None, y_max=None, file_path=None):
    """이동평균 분석 시각화"""
    try:
        # 입력 데이터 검증
        if not ma_results or len(ma_results) == 0:
            logger.warning("No moving average results to plot")
            return None, None
            
        # ma_results 형식: {'ma5': [{'date': '...', 'prediction': X, 'actual': Y, 'ma': Z}, ...], 'ma10': [...]}
        windows = sorted(ma_results.keys())
        
        if len(windows) == 0:
            logger.warning("No moving average windows found")
            return None, None
        
        # 유효한 윈도우 필터링
        valid_windows = []
        for window_key in windows:
            if window_key in ma_results and ma_results[window_key] and len(ma_results[window_key]) > 0:
                valid_windows.append(window_key)
        
        if len(valid_windows) == 0:
            logger.warning("No valid moving average data found")
            return None, None
        
        fig = plt.figure(figsize=(12, max(4, 4 * len(valid_windows))))
        
        if isinstance(sequence_start_date, str):
            title = f"{title_prefix} Starting {sequence_start_date}"
        else:
            title = f"{title_prefix} Starting {sequence_start_date.strftime('%Y-%m-%d')}"
            
        fig.suptitle(title, fontsize=16)
        
        for idx, window_key in enumerate(valid_windows):
            window_num = window_key.replace('ma', '')
            ax = fig.add_subplot(len(valid_windows), 1, idx+1)
            
            window_data = ma_results[window_key]
            
            # 데이터 검증
            if not window_data or len(window_data) == 0:
                ax.text(0.5, 0.5, f"No data for {window_key}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # 날짜, 예측, 실제값, MA 추출
            dates = []
            predictions = []
            actuals = []
            ma_preds = []
            
            for item in window_data:
                try:
                    # 안전한 데이터 추출
                    if isinstance(item['date'], str):
                        dates.append(pd.to_datetime(item['date']))
                    else:
                        dates.append(item['date'])
                    
                    # None 값 처리
                    predictions.append(item.get('prediction', 0))
                    actuals.append(item.get('actual', None))
                    ma_preds.append(item.get('ma', None))
                except Exception as e:
                    logger.warning(f"Error processing MA data item: {str(e)}")
                    continue
            
            if len(dates) == 0:
                ax.text(0.5, 0.5, f"No valid data for {window_key}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
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
        
        # 📁 저장 디렉토리 설정 (파일별 캐시 디렉토리 사용)
        if save_prefix is None:
            try:
                cache_dirs = get_file_cache_dirs(file_path)
                save_dir = cache_dirs['ma_plots']
            except Exception as e:
                logger.warning(f"Could not get cache directories for MA plots: {str(e)}")
                save_dir = Path("temp_ma_plots")
                save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = Path(save_prefix)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(sequence_start_date, str):
            date_str = pd.to_datetime(sequence_start_date).strftime('%Y%m%d')
        else:
            date_str = sequence_start_date.strftime('%Y%m%d')
            
        filename = save_dir / f"ma_analysis_{date_str}.png"
        
        # 이미지를 메모리에 저장
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png', dpi=300)
        img_buf.seek(0)
        
        # Base64로 인코딩
        img_str = base64.b64encode(img_buf.read()).decode('utf-8')
        
        # 파일 저장
        plt.savefig(str(filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Moving Average graph saved: {filename}")
        return str(filename), img_str
        
    except Exception as e:
        logger.error(f"Error in moving average visualization: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def compute_performance_metrics_improved(validation_data, start_day_value):
    """
    검증 데이터만을 사용한 성능 지표 계산
    """
    try:
        if not validation_data or len(validation_data) < 1:
            logger.info("No validation data available - this is normal for pure future predictions")
            return None
        
        # 검증 데이터에서 값 추출
        actual_vals = [start_day_value] + [item['actual'] for item in validation_data]
        pred_vals = [start_day_value] + [item['prediction'] for item in validation_data]
        
        # F1 점수 계산
        f1, f1_report = calculate_f1_score(actual_vals, pred_vals)
        direction_accuracy = calculate_direction_accuracy(actual_vals, pred_vals)
        weighted_score, max_score = calculate_direction_weighted_score(actual_vals[1:], pred_vals[1:])
        weighted_score_pct = (weighted_score / max_score) * 100 if max_score > 0 else 0.0
        mape = calculate_mape(actual_vals[1:], pred_vals[1:])
        
        # 코사인 유사도
        cosine_similarity = None
        if len(actual_vals) > 1:
            diff_actual = np.diff(actual_vals)
            diff_pred = np.diff(pred_vals)
            norm_actual = np.linalg.norm(diff_actual)
            norm_pred = np.linalg.norm(diff_pred)
            if norm_actual > 0 and norm_pred > 0:
                cosine_similarity = np.dot(diff_actual, diff_pred) / (norm_actual * norm_pred)
        
        return {
            'f1': float(f1),
            'accuracy': float(direction_accuracy),
            'mape': float(mape),
            'weighted_score': float(weighted_score_pct),
            'cosine_similarity': float(cosine_similarity) if cosine_similarity is not None else None,
            'f1_report': f1_report,
            'validation_points': len(validation_data)
        }
        
    except Exception as e:
        logger.error(f"Error computing improved metrics: {str(e)}")
        return None

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
        # 입력 데이터 검증
        if not predictions or len(predictions) == 0:
            logger.warning("No predictions provided for moving average calculation")
            return {}
            
        if historical_data is None or historical_data.empty:
            logger.warning("No historical data provided for moving average calculation")
            return {}
            
        if target_col not in historical_data.columns:
            logger.warning(f"Target column {target_col} not found in historical data")
            return {}
        
        results = {}
        
        # 예측 데이터를 DataFrame으로 변환 및 정렬
        try:
            pred_df = pd.DataFrame(predictions) if not isinstance(predictions, pd.DataFrame) else predictions.copy()
            
            # Date 컬럼 검증
            if 'Date' not in pred_df.columns:
                logger.error("Date column not found in predictions")
                return {}
                
            pred_df['Date'] = pd.to_datetime(pred_df['Date'])
            pred_df = pred_df.sort_values('Date')
            
            # Prediction 컬럼 검증
            if 'Prediction' not in pred_df.columns:
                logger.error("Prediction column not found in predictions")
                return {}
                
        except Exception as e:
            logger.error(f"Error processing prediction data: {str(e)}")
            return {}
        
        # 예측 시작일 확인
        prediction_start_date = pred_df['Date'].min()
        logger.info(f"MA calculation - prediction start date: {prediction_start_date}")
        
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
            logger.info(f"MA{window} calculated: {len(window_results)} data points")
        
        logger.info(f"Moving average calculation completed with {len(results)} windows")
        return results
        
    except Exception as e:
        logger.error(f"Error calculating moving averages with history: {str(e)}")
        logger.error(traceback.format_exc())
        return {}

# 2. 여러 날짜에 대한 누적 예측을 수행하는 함수 추가
def run_accumulated_predictions_with_save(file_path, start_date, end_date=None, save_to_csv=True, use_saved_data=True):
    """
    시작 날짜부터 종료 날짜까지 각 날짜별로 예측을 수행하고 결과를 누적합니다. (수정됨)
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
        prediction_state['accumulated_consistency_scores'] = {}
        prediction_state['current_file'] = file_path  # ✅ 현재 파일 경로 설정
        
        logger.info(f"Running accumulated predictions from {start_date} to {end_date}")

        # 입력 날짜를 datetime 객체로 변환
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if end_date is not None and isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # 저장된 데이터 활용 옵션이 켜져 있으면 먼저 CSV에서 로드 시도
        loaded_predictions = []
        if use_saved_data:
            logger.info("🔍 [CACHE] Attempting to load existing predictions from CSV files...")
            
            # 🔧 인덱스 파일이 없으면 기존 파일들로부터 재생성
            cache_dirs = get_file_cache_dirs(file_path)
            predictions_index_file = cache_dirs['predictions'] / 'predictions_index.csv'
            
            if not predictions_index_file.exists():
                logger.warning("⚠️ [CACHE] predictions_index.csv not found, attempting to rebuild from existing files...")
                if rebuild_predictions_index_from_existing_files():
                    logger.info("✅ [CACHE] Successfully rebuilt predictions index")
                else:
                    logger.warning("⚠️ [CACHE] Failed to rebuild predictions index")
            
            loaded_predictions = load_accumulated_predictions_from_csv(start_date, end_date, file_path=file_path)  # ✅ 파일 경로 추가
            logger.info(f"📦 [CACHE] Successfully loaded {len(loaded_predictions)} predictions from CSV cache")
            if len(loaded_predictions) > 0:
                logger.info(f"💡 [CACHE] Using cached predictions will significantly speed up processing!")

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

        # 이미 로드된 예측 결과들을 날짜별 딕셔너리로 변환
        loaded_by_date = {}
        for pred in loaded_predictions:
            loaded_by_date[pred['date']] = pred

        # ✅ 캐시 활용 통계 초기화
        cache_statistics = {
            'total_dates': 0,
            'cached_dates': 0,
            'new_predictions': 0,
            'cache_hit_rate': 0.0
        }

        all_predictions = []
        accumulated_interval_scores = {}

        # 각 날짜별 예측 수행 또는 로드
        for i, current_date in enumerate(available_dates):
            current_date_str = format_date(current_date)
            cache_statistics['total_dates'] += 1
            
            logger.info(f"Processing date {i+1}/{total_dates}: {current_date_str}")
            
            # 이미 로드된 데이터가 있으면 사용
            if current_date_str in loaded_by_date:
                cache_statistics['cached_dates'] += 1  # ✅ 캐시 사용 시 카운터 증가
                logger.info(f"⚡ [CACHE] Using cached prediction for {current_date_str} (skipping computation)")
                date_result = loaded_by_date[current_date_str]
                
                # 🔧 캐시된 metrics 안전성 처리
                metrics = date_result.get('metrics')
                if not metrics or not isinstance(metrics, dict):
                    logger.warning(f"⚠️ [CACHE] Invalid metrics for {current_date_str}, using defaults")
                    metrics = {'f1': 0.0, 'accuracy': 0.0, 'mape': 0.0, 'weighted_score': 0.0}
                
                # 누적 성능 지표 업데이트
                accumulated_metrics['f1'] += metrics.get('f1', 0.0)
                accumulated_metrics['accuracy'] += metrics.get('accuracy', 0.0)
                accumulated_metrics['mape'] += metrics.get('mape', 0.0)
                accumulated_metrics['weighted_score'] += metrics.get('weighted_score', 0.0)
                accumulated_metrics['total_predictions'] += 1
                
            else:
                # 새로운 예측 수행
                cache_statistics['new_predictions'] += 1
                logger.info(f"🚀 [COMPUTE] Running new prediction for {current_date_str} (not in cache)")
                try:
                    # ✅ 누적 예측에서도 모든 새 예측을 저장하도록 보장
                    results = generate_predictions_with_save(df, current_date, save_to_csv=True, file_path=file_path)
                    
                    # 예측 데이터 타입 안전 확인
                    predictions = results.get('predictions_flat', results.get('predictions', []))
                    
                    # 예측 데이터가 중첩된 딕셔너리 구조인 경우 처리
                    if isinstance(predictions, dict):
                        if 'future' in predictions:
                            predictions = predictions['future']
                        elif 'predictions' in predictions:
                            predictions = predictions['predictions']
                    
                    if not predictions or not isinstance(predictions, list):
                        logger.warning(f"No valid predictions found for {current_date_str}: {type(predictions)}")
                        continue
                        
                    # 실제 예측한 영업일 수 계산 (안전한 방식)
                    actual_business_days = 0
                    try:
                        for p in predictions:
                            # p가 딕셔너리인지 확인
                            if isinstance(p, dict):
                                date_key = p.get('Date') or p.get('date')
                                is_synthetic = p.get('is_synthetic', False)
                                if date_key and not is_synthetic:
                                    actual_business_days += 1
                            else:
                                logger.warning(f"Prediction item is not dict for {current_date_str}: {type(p)}")
                    except Exception as calc_error:
                        logger.error(f"Error calculating business days: {str(calc_error)}")
                        actual_business_days = len(predictions)  # 기본값
                    
                    metrics = results.get('metrics', {})
                    if not metrics:
                        # 메트릭이 없으면 기본값 설정
                        metrics = {'f1': 0.0, 'accuracy': 0.0, 'mape': 0.0, 'weighted_score': 0.0}
                    
                    # 누적 성능 지표 업데이트
                    accumulated_metrics['f1'] += metrics.get('f1', 0.0)
                    accumulated_metrics['accuracy'] += metrics.get('accuracy', 0.0)
                    accumulated_metrics['mape'] += metrics.get('mape', 0.0)
                    accumulated_metrics['weighted_score'] += metrics.get('weighted_score', 0.0)
                    accumulated_metrics['total_predictions'] += 1

                    # 안전한 데이터 구조 생성
                    safe_predictions = predictions if isinstance(predictions, list) else []
                    safe_interval_scores = results.get('interval_scores', {})
                    if not isinstance(safe_interval_scores, dict):
                        safe_interval_scores = {}
                    
                    date_result = {
                        'date': current_date_str,
                        'predictions': safe_predictions,
                        'metrics': metrics,
                        'interval_scores': safe_interval_scores,
                        'actual_business_days': actual_business_days,
                        'next_semimonthly_period': results.get('next_semimonthly_period'),
                        'original_interval_scores': safe_interval_scores,
                        'ma_results': results.get('ma_results', {}),  # 🔑 이동평균 데이터 추가
                        'attention_data': results.get('attention_data', {})  # 🔑 Attention 데이터 추가
                    }
                    
                except Exception as e:
                    logger.error(f"Error in prediction for date {current_date}: {str(e)}")
                    logger.error(traceback.format_exc())
                    continue

            # 구간 점수 누적 처리 (안전한 방식)
            interval_scores = date_result.get('interval_scores', {})
            if isinstance(interval_scores, dict):
                for interval in interval_scores.values():
                    if not interval or not isinstance(interval, dict) or 'days' not in interval or interval['days'] is None:
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

            all_predictions.append(date_result)
            prediction_state['prediction_progress'] = 10 + int(90 * (i + 1) / total_dates)

        # 평균 성능 지표 계산
        if accumulated_metrics['total_predictions'] > 0:
            count = accumulated_metrics['total_predictions']
            accumulated_metrics['f1'] /= count
            accumulated_metrics['accuracy'] /= count
            accumulated_metrics['mape'] /= count
            accumulated_metrics['weighted_score'] /= count

        # 예측 신뢰도 계산
        logger.info("Calculating prediction consistency scores...")
        unique_periods = set()
        for pred in all_predictions:
            if 'next_semimonthly_period' in pred and pred['next_semimonthly_period']:
                unique_periods.add(pred['next_semimonthly_period'])
        
        accumulated_consistency_scores = {}
        for period in unique_periods:
            try:
                consistency_data = calculate_prediction_consistency(all_predictions, period)
                accumulated_consistency_scores[period] = consistency_data
                logger.info(f"Consistency score for {period}: {consistency_data.get('consistency_score', 'N/A')}")
            except Exception as e:
                logger.error(f"Error calculating consistency for period {period}: {str(e)}")

        # accumulated_interval_scores 처리
        accumulated_scores_list = [
            v for v in accumulated_interval_scores.values()
            if v is not None and isinstance(v, dict) and 'days' in v and v['days'] is not None
        ]
        accumulated_scores_list.sort(key=lambda x: x['score'], reverse=True)

        accumulated_purchase_reliability, debug_info = calculate_accumulated_purchase_reliability_with_debug(all_predictions)
        
        # ✅ 캐시 활용률 계산
        cache_statistics['cache_hit_rate'] = (cache_statistics['cached_dates'] / cache_statistics['total_dates'] * 100) if cache_statistics['total_dates'] > 0 else 0.0
        logger.info(f"🎯 [CACHE] Final statistics: {cache_statistics['cached_dates']}/{cache_statistics['total_dates']} cached ({cache_statistics['cache_hit_rate']:.1f}%), {cache_statistics['new_predictions']} new predictions computed")
        
        # 결과 저장
        prediction_state['accumulated_predictions'] = all_predictions
        prediction_state['accumulated_metrics'] = accumulated_metrics
        prediction_state['prediction_dates'] = [p['date'] for p in all_predictions]
        prediction_state['accumulated_interval_scores'] = accumulated_scores_list
        prediction_state['accumulated_consistency_scores'] = accumulated_consistency_scores
        prediction_state['accumulated_purchase_reliability'] = accumulated_purchase_reliability
        prediction_state['accumulated_purchase_debug'] = debug_info
        prediction_state['cache_statistics'] = cache_statistics  # ✅ 캐시 통계 추가

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
        prediction_state['accumulated_consistency_scores'] = {}

# 3. 백그라운드에서 누적 예측을 수행하는 함수
def background_accumulated_prediction(file_path, start_date, end_date=None):
    """백그라운드에서 누적 예측을 수행하는 함수"""
    thread = Thread(target=run_accumulated_predictions_with_save, args=(file_path, start_date, end_date))
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
        
        # 보고서 파일 이름 생성 - 파일별 캐시 디렉토리 사용
        start_date = all_preds[0]['date']
        end_date = all_preds[-1]['date']
        try:
            cache_dirs = get_file_cache_dirs()
            report_dir = cache_dirs['predictions']
            report_filename = os.path.join(report_dir, f"accumulated_report_{start_date}_to_{end_date}.txt")
        except Exception as e:
            logger.warning(f"Could not get cache directories for accumulated report: {str(e)}")
            report_filename = f"accumulated_report_{start_date}_to_{end_date}.txt"
        
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
        
        # 파일로 저장 - 파일별 캐시 디렉토리 사용
        try:
            cache_dirs = get_file_cache_dirs()
            plots_dir = cache_dirs['plots']
            filename = os.path.join(plots_dir, 'accumulated_metrics.png')
        except Exception as e:
            logger.warning(f"Could not get cache directories for accumulated metrics: {str(e)}")
            filename = 'accumulated_metrics.png'
        
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

# generate_predictions 함수 수정 (간단하고 정확한 버전)

def generate_predictions(df, current_date, predict_window=23, features=None, target_col='MOPJ', file_path=None):
    """
    개선된 예측 수행 함수 - 예측 시작일의 반월 기간 하이퍼파라미터 사용
    🔑 데이터 누출 방지: current_date 이후의 실제값은 사용하지 않음
    """
    try:
        # 디바이스 설정
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # 현재 날짜가 문자열이면 datetime으로 변환
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        # 현재 날짜 검증 (데이터 기준일)
        if current_date not in df.index:
            closest_date = df.index[df.index <= current_date][-1]
            logger.warning(f"Current date {current_date} not found in dataframe. Using closest date: {closest_date}")
            current_date = closest_date
        
        # 예측 시작일 계산
        prediction_start_date = current_date + pd.Timedelta(days=1)
        while prediction_start_date.weekday() >= 5 or is_holiday(prediction_start_date):
            prediction_start_date += pd.Timedelta(days=1)
        
        # 반월 기간 계산
        data_semimonthly_period = get_semimonthly_period(current_date)
        prediction_semimonthly_period = get_semimonthly_period(prediction_start_date)
        
        # ✅ 핵심 수정: 예측 시작일 기준으로 다음 반월 계산
        next_semimonthly_period = get_next_semimonthly_period(prediction_start_date)
        
        logger.info(f"🎯 Prediction Setup:")
        logger.info(f"  📅 Data base date: {current_date} (period: {data_semimonthly_period})")
        logger.info(f"  🚀 Prediction start date: {prediction_start_date} (period: {prediction_semimonthly_period})")
        logger.info(f"  🎯 Purchase interval target period: {next_semimonthly_period}")
        
        # 23일치 예측을 위한 날짜 생성
        all_business_days = get_next_n_business_days(current_date, df, predict_window)
        
        # ✅ 핵심 수정: 예측 시작일 기준으로 구매 구간 계산
        semimonthly_business_days, purchase_target_period = get_next_semimonthly_dates(prediction_start_date, df)
        
        logger.info(f"  📊 Total predictions: {len(all_business_days)} days")
        logger.info(f"  🛒 Purchase target period: {purchase_target_period}")
        logger.info(f"  📈 Purchase interval business days: {len(semimonthly_business_days)}")
        
        if not all_business_days:
            raise ValueError(f"No future business days found after {current_date}")

        # ✅ 핵심 수정: 날짜별로 다른 학습 데이터 사용 보장
        historical_data = df[df.index <= current_date].copy()
        
        logger.info(f"  📊 Training data: {len(historical_data)} records up to {format_date(current_date)}")
        logger.info(f"  📊 Training data range: {format_date(historical_data.index.min())} ~ {format_date(historical_data.index.max())}")
        
        # 최소 데이터 요구사항 확인
        if len(historical_data) < 50:
            raise ValueError(f"Insufficient training data: {len(historical_data)} records (minimum 50 required)")
        
        if features is None:
            selected_features, _ = select_features_from_groups(
                historical_data, 
                variable_groups,
                target_col=target_col,
                vif_threshold=50.0,
                corr_threshold=0.8
            )
        else:
            selected_features = features
            
        if target_col not in selected_features:
            selected_features.append(target_col)
        
        logger.info(f"  🔧 Selected features ({len(selected_features)}): {selected_features}")
        
        # ✅ 핵심 수정: 날짜별 다른 스케일링 보장
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(historical_data[selected_features])
        target_col_idx = selected_features.index(target_col)
        
        logger.info(f"  ⚖️  Scaler fitted on data up to {format_date(current_date)}")
        logger.info(f"  📊 Scaled data shape: {scaled_data.shape}")
        
        # ✅ 핵심: 예측 시작일의 반월 기간 하이퍼파라미터 사용
        optimized_params = optimize_hyperparameters_semimonthly_kfold(
            train_data=scaled_data,
            input_size=len(selected_features),
            target_col_idx=target_col_idx,
            device=device,
            current_period=prediction_semimonthly_period,  # ✅ 예측 시작일의 반월 기간
            file_path=file_path,  # 🔑 파일 경로 전달
            n_trials=30,
            k_folds=5,
            use_cache=True
        )
        
        logger.info(f"✅ Using hyperparameters for prediction start period: {prediction_semimonthly_period}")
        
        # ✅ 핵심 수정: 모델 학습 시 현재 날짜 기준으로 데이터 분할 보장
        logger.info(f"  🚀 Training model with data up to {format_date(current_date)}")
        model, model_scaler, model_target_col_idx = train_model(
            selected_features,
            target_col,
            current_date,
            historical_data,
            device,
            optimized_params
        )
        
        # 스케일러 일관성 확인
        if model_target_col_idx != target_col_idx:
            logger.warning(f"Target column index mismatch: {model_target_col_idx} vs {target_col_idx}")
            target_col_idx = model_target_col_idx
        
        logger.info(f"  ✅ Model trained successfully for prediction starting {format_date(prediction_start_date)}")
        
        # ✅ 핵심 수정: 예측 데이터 준비 시 날짜별 다른 시퀀스 보장
        seq_len = optimized_params['sequence_length']
        current_idx = df.index.get_loc(current_date)
        start_idx = max(0, current_idx - seq_len + 1)
        
        # 시퀀스 데이터 추출 (current_date까지만!)
        sequence = df.iloc[start_idx:current_idx+1][selected_features].values
        
        logger.info(f"  📊 Sequence data: {sequence.shape} from {format_date(df.index[start_idx])} to {format_date(current_date)}")
        
        # 모델에서 반환된 스케일러 사용 (일관성 보장)
        sequence = model_scaler.transform(sequence)
        prev_value = sequence[-1, target_col_idx]
        
        logger.info(f"  📈 Previous value (scaled): {prev_value:.4f}")
        logger.info(f"  📊 Sequence length used: {len(sequence)} (required: {seq_len})")
        
        # 예측 수행
        future_predictions = []  # 미래 예측 (실제값 없음)
        validation_data = []     # 검증 데이터 (실제값 있음)
        
        with torch.no_grad():
            # 23영업일 전체에 대해 예측 수행
            max_pred_days = min(predict_window, len(all_business_days))
            current_sequence = sequence.copy()
            
            # 텐서로 변환
            X = torch.FloatTensor(current_sequence).unsqueeze(0).to(device)
            prev_tensor = torch.FloatTensor([prev_value]).to(device)
            
            # 전체 시퀀스 예측
            pred = model(X, prev_tensor).cpu().numpy()[0]
            
            # ✅ 핵심 수정: 각 날짜별 예측 생성 (데이터 누출 방지)
            for j, pred_date in enumerate(all_business_days[:max_pred_days]):
                # ✅ 스케일 역변환 시 일관된 스케일러 사용
                dummy_matrix = np.zeros((1, len(selected_features)))
                dummy_matrix[0, target_col_idx] = pred[j]
                pred_value = model_scaler.inverse_transform(dummy_matrix)[0, target_col_idx]
                
                # 예측값 검증 및 정리
                if np.isnan(pred_value) or np.isinf(pred_value):
                    logger.warning(f"Invalid prediction value for {pred_date}: {pred_value}, skipping")
                    continue
                
                pred_value = float(pred_value)
                
                # 기본 예측 정보
                prediction_item = {
                    'date': format_date(pred_date, '%Y-%m-%d'),
                    'prediction': pred_value,
                    'prediction_from': format_date(current_date, '%Y-%m-%d'),
                    'day_offset': j + 1,
                    'is_business_day': pred_date.weekday() < 5 and not is_holiday(pred_date),
                    'is_synthetic': pred_date not in df.index,
                    'semimonthly_period': data_semimonthly_period,
                    'next_semimonthly_period': next_semimonthly_period
                }
                
                # ✅ 실제 데이터 마지막 날짜 확인 (검증용)
                last_data_date = df.index.max()
                
                # ✅ 검증 조건: 예측 날짜가 실제 데이터 범위 내에 있고, current_date 이후라면 검증용으로 사용
                if (pred_date in df.index and 
                    pd.notna(df.loc[pred_date, target_col]) and 
                    pred_date <= last_data_date):  # 🔑 실제 데이터 범위 내에서 검증 허용
                    
                    actual_value = float(df.loc[pred_date, target_col])
                    
                    if not (np.isnan(actual_value) or np.isinf(actual_value)):
                        validation_item = {
                            **prediction_item,
                            'actual': actual_value,
                            'error': abs(pred_value - actual_value),
                            'error_pct': abs(pred_value - actual_value) / actual_value * 100 if actual_value != 0 else 0.0
                        }
                        validation_data.append(validation_item)
                        
                        # 📊 검증 타입 구분 로그
                        if pred_date <= current_date:
                            logger.debug(f"  ✅ Training validation: {format_date(pred_date)} - Pred: {pred_value:.2f}, Actual: {actual_value:.2f}")
                        else:
                            logger.debug(f"  🎯 Test validation: {format_date(pred_date)} - Pred: {pred_value:.2f}, Actual: {actual_value:.2f}")
                elif pred_date > last_data_date:
                    logger.debug(f"  🔮 Future: {format_date(pred_date)} - Pred: {pred_value:.2f} (no actual - beyond data)")
                
                future_predictions.append(prediction_item)
        
        # 📊 검증 데이터 통계
        training_validation = len([v for v in validation_data if pd.to_datetime(v['date']) <= current_date])
        test_validation = len([v for v in validation_data if pd.to_datetime(v['date']) > current_date])
        
        logger.info(f"📊 Prediction Results:")
        logger.info(f"  📈 Total predictions: {len(future_predictions)}")
        logger.info(f"  ✅ Training validation (≤ {format_date(current_date)}): {training_validation}")
        logger.info(f"  🎯 Test validation (> {format_date(current_date)}): {test_validation}")
        logger.info(f"  📋 Total validation points: {len(validation_data)}")
        logger.info(f"  🔮 Pure future predictions (> {format_date(df.index.max())}): {len(future_predictions) - len(validation_data)}")
        
        if len(validation_data) == 0:
            logger.info("  ℹ️  Pure future prediction - no validation data available")
        
        # ✅ 구간 평균 및 점수 계산 - 올바른 구매 대상 기간 사용
        temp_predictions_for_interval = []
        for pred in future_predictions:
            pred_date = pd.to_datetime(pred['date'])
            if pred_date in semimonthly_business_days:  # 이제 올바른 다음 반월 날짜들
                temp_predictions_for_interval.append({
                    'Date': pred_date,
                    'Prediction': pred['prediction']
                })
        
        logger.info(f"  🛒 Predictions for interval calculation: {len(temp_predictions_for_interval)} (target period: {purchase_target_period})")
        
        interval_averages, interval_scores, analysis_info = calculate_interval_averages_and_scores(
            temp_predictions_for_interval, 
            semimonthly_business_days
        )

        # 최종 구매 구간 결정
        best_interval = decide_purchase_interval(interval_scores)

        # 성능 메트릭 계산 (검증 데이터가 있을 때만)
        metrics = None
        if validation_data:
            start_day_value = df.loc[current_date, target_col]
            if not (pd.isna(start_day_value) or np.isnan(start_day_value) or np.isinf(start_day_value)):
                try:
                    temp_df_for_metrics = pd.DataFrame([
                        {
                            'Date': pd.to_datetime(item['date']),
                            'Prediction': item['prediction'],
                            'Actual': item['actual']
                        } for item in validation_data
                    ])
                    
                    if not temp_df_for_metrics.empty:
                        metrics = compute_performance_metrics_improved(temp_df_for_metrics, start_day_value)
                        logger.info(f"  📊 Computed metrics from {len(validation_data)} validation points")
                    else:
                        logger.info("  ⚠️  No valid data for metrics computation")
                except Exception as e:
                    logger.error(f"Error computing metrics: {str(e)}")
                    metrics = None
            else:
                logger.warning("Invalid start_day_value for metrics computation")
        else:
            logger.info("  ℹ️  No validation data available - pure future prediction")
        
        # ✅ 이동평균 계산 시 실제값도 포함 (검증 데이터가 있는 경우)
        temp_predictions_for_ma = []
        for pred in future_predictions:
            pred_date = pd.to_datetime(pred['date'])
            actual_val = None
            
            # 실제 데이터가 존재하는 날짜면 실제값 설정
            if (pred_date in df.index and 
                pd.notna(df.loc[pred_date, target_col]) and 
                pred_date <= df.index.max()):
                actual_val = float(df.loc[pred_date, target_col])
            
            temp_predictions_for_ma.append({
                'Date': pred_date,
                'Prediction': pred['prediction'],
                'Actual': actual_val
            })
        
        logger.info(f"  📈 Calculating moving averages with historical data up to {format_date(current_date)}")
        ma_results = calculate_moving_averages_with_history(
            temp_predictions_for_ma, 
            historical_data,  # 이미 current_date까지로 필터링됨
            target_col=target_col
        )
        
        # 특성 중요도 분석
        attention_data = None
        try:
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            prev_tensor = torch.FloatTensor([float(prev_value)]).to(device)
            
            attention_file, attention_img, feature_importance = visualize_attention_weights(
                model, sequence_tensor, prev_tensor, prediction_start_date, selected_features
            )
            
            attention_data = {
                'image': attention_img,
                'file_path': attention_file,
                'feature_importance': feature_importance
            }
        except Exception as e:
            logger.error(f"Error in attention analysis: {str(e)}")
        
        # 시각화 생성
        plots = {
            'basic_plot': {'file': None, 'image': None},
            'ma_plot': {'file': None, 'image': None}
        }

        start_day_value = None
        if current_date in df.index and not pd.isna(df.loc[current_date, target_col]):
            start_day_value = df.loc[current_date, target_col]

        # 📊 시각화용 데이터 준비 - 실제값 포함
        temp_df_for_plot_data = []
        for item in future_predictions:
            pred_date = pd.to_datetime(item['date'])
            actual_val = None
            
            # 실제 데이터가 존재하는 날짜면 실제값 설정
            if (pred_date in df.index and 
                pd.notna(df.loc[pred_date, target_col]) and 
                pred_date <= df.index.max()):
                actual_val = float(df.loc[pred_date, target_col])
            
            temp_df_for_plot_data.append({
                'Date': pred_date,
                'Prediction': item['prediction'],
                'Actual': actual_val
            })
        
        temp_df_for_plot = pd.DataFrame(temp_df_for_plot_data)

        if metrics:
            f1_score = metrics['f1']
            accuracy = metrics['accuracy']
            mape = metrics['mape']
            weighted_score = metrics['weighted_score']
            visualization_type = "with validation data"
        else:
            f1_score = accuracy = mape = weighted_score = 0.0
            visualization_type = "future prediction only"

        if start_day_value is not None and not temp_df_for_plot.empty:
            try:
                # ✅ current_date 전달 추가
                basic_plot_file, basic_plot_img = plot_prediction_basic(
                    temp_df_for_plot, 
                    prediction_start_date, 
                    start_day_value,
                    f1_score,
                    accuracy,
                    mape,
                    weighted_score,
                    current_date=current_date,  # 🔑 추가
                    save_prefix=None,  # 파일별 캐시 디렉토리 자동 사용
                    title_prefix="Future Price Prediction" if not validation_data else "Prediction with Validation",
                    file_path=file_path  # 🔑 추가
                )
                
                ma_plot_file, ma_plot_img = plot_moving_average_analysis(
                    ma_results, 
                    prediction_start_date,
                    save_prefix=None,  # 파일별 캐시 디렉토리 자동 사용
                    title_prefix="Moving Average Analysis (Future Prediction)" if not validation_data else "Moving Average Analysis",
                    file_path=file_path  # 🔑 추가
                )
                
                plots = {
                    'basic_plot': {'file': basic_plot_file, 'image': basic_plot_img},
                    'ma_plot': {'file': ma_plot_file, 'image': ma_plot_img}
                }
                
                logger.info(f"  🎨 Visualization generated ({visualization_type}) - {len(future_predictions)} predictions")
                
            except Exception as e:
                logger.error(f"Error generating visualization: {str(e)}")
                plots = {
                    'basic_plot': {'file': None, 'image': None},
                    'ma_plot': {'file': None, 'image': None}
                }
        else:
            logger.warning(f"Cannot generate visualization: start_value={start_day_value}, predictions={len(temp_df_for_plot)}")

        # 결과 리포트 생성 - 파일별 캐시 디렉토리 사용
        try:
            cache_dirs = get_file_cache_dirs(file_path)
            report_dir = cache_dirs['predictions']  # predictions 디렉토리에 저장
            report_filename = os.path.join(report_dir, f"prediction_report_{format_date(prediction_start_date, '%Y%m%d')}.txt")
        except Exception as e:
            logger.warning(f"Could not get cache directories for report: {str(e)}")
            report_filename = f"prediction_report_{format_date(prediction_start_date, '%Y%m%d')}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"===== Prediction Report for {format_date(prediction_start_date)} =====\n\n")
            
            f.write(f"Data Base Date: {format_date(current_date)}\n")
            f.write(f"Prediction Start Date: {format_date(prediction_start_date)}\n")
            f.write(f"Data Semimonthly Period: {data_semimonthly_period}\n")
            f.write(f"Prediction Semimonthly Period: {prediction_semimonthly_period}\n")
            f.write(f"Hyperparameters Used: {prediction_semimonthly_period}\n")
            f.write(f"Next Semimonthly Period: {next_semimonthly_period}\n")
            f.write(f"Total Predictions: {len(future_predictions)}\n")
            f.write(f"Validation Points: {len(validation_data)}\n\n")
            
            if metrics:
                f.write("Performance Metrics:\n")
                f.write(f"- F1 Score: {metrics['f1']:.4f}\n")
                f.write(f"- Direction Accuracy: {metrics['accuracy']:.2f}%\n")
                f.write(f"- MAPE: {metrics['mape']:.2f}%\n")
                f.write(f"- Weighted Score: {metrics['weighted_score']:.2f}%\n")
                if metrics['cosine_similarity'] is not None:
                    f.write(f"- Cosine Similarity: {metrics['cosine_similarity']:.4f}\n")
                f.write("\n")
            else:
                f.write("Performance Metrics: Not available (pure future prediction)\n\n")
            
            f.write(f"Selected Features ({len(selected_features)}):\n")
            for feature in selected_features:
                f.write(f"- {feature}\n")
            f.write("\n")
            
            if best_interval:
                f.write("Best Purchase Interval:\n")
                f.write(f"- Start Date: {best_interval['start_date']}\n")
                f.write(f"- End Date: {best_interval['end_date']}\n")
                f.write(f"- Duration: {best_interval['days']} days\n")
                f.write(f"- Average Price: {best_interval['avg_price']:.2f}\n")
                f.write(f"- Score: {best_interval['score']}\n")
                f.write(f"- Selection Reason: {best_interval.get('selection_reason', '')}\n\n")
        
        # 첫 번째 예측 날짜를 실제 기준 시점으로 설정
        first_prediction_date = all_business_days[0] if all_business_days else prediction_start_date

        # 결과에 올바른 구매 대상 기간 정보 추가
        return {
            'predictions': {
                'future': future_predictions,
                'validation': validation_data
            },
            
            'predictions_flat': future_predictions,
            
            'summary': {
                'total_predictions': len(future_predictions),
                'validation_points': len(validation_data),
                'prediction_start_date': format_date(prediction_start_date, '%Y-%m-%d'),
                'prediction_end_date': format_date(all_business_days[-1], '%Y-%m-%d') if all_business_days else None,
                'data_end_date': format_date(current_date, '%Y-%m-%d'),
                'is_pure_future_prediction': len(validation_data) == 0,
                'hyperparameter_period_used': prediction_semimonthly_period,
                'purchase_target_period': purchase_target_period  # ✅ 추가
            },
            
            'interval_scores': interval_scores,
            'interval_averages': interval_averages,
            'best_interval': best_interval,
            'ma_results': ma_results,
            'metrics': metrics,
            'selected_features': selected_features,
            'attention_data': attention_data,
            'plots': plots,
            'report_file': report_filename,
            'current_date': format_date(prediction_start_date, '%Y-%m-%d'),
            'data_end_date': format_date(current_date, '%Y-%m-%d'),
            'semimonthly_period': data_semimonthly_period,
            'next_semimonthly_period': purchase_target_period,  # ✅ 수정: 올바른 구매 대상 기간
            'prediction_semimonthly_period': prediction_semimonthly_period,
            'hyperparameter_period_used': prediction_semimonthly_period,
            'purchase_target_period': purchase_target_period  # ✅ 추가
        }
        
    except Exception as e:
        logger.error(f"Error in prediction generation: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

def generate_predictions_compatible(df, current_date, predict_window=23, features=None, target_col='MOPJ'):
    """
    기존 프론트엔드와 호환되는 예측 함수
    (새로운 구조 + 기존 형태 변환)
    """
    try:
        # 새로운 generate_predictions 함수 실행
        new_results = generate_predictions(df, current_date, predict_window, features, target_col)
        
        # 기존 형태로 변환
        if isinstance(new_results.get('predictions'), dict):
            # 새로운 구조인 경우
            future_predictions = new_results['predictions']['future']
            validation_data = new_results['predictions']['validation']
            
            # future와 validation을 합쳐서 기존 형태로 변환
            all_predictions = future_predictions + validation_data
        else:
            # 기존 구조인 경우
            all_predictions = new_results.get('predictions_flat', new_results.get('predictions', []))
        
        # 기존 필드명으로 변환
        compatible_predictions = convert_to_legacy_format(all_predictions)
        
        # 결과에 기존 형태 추가
        new_results['predictions'] = compatible_predictions  # 기존 호환성
        new_results['predictions_new'] = new_results.get('predictions')  # 새로운 구조도 유지
        
        logger.info(f"Generated {len(compatible_predictions)} compatible predictions")
        
        return new_results
        
    except Exception as e:
        logger.error(f"Error in compatible prediction generation: {str(e)}")
        raise e

def generate_predictions_with_save(df, current_date, predict_window=23, features=None, target_col='MOPJ', save_to_csv=True, file_path=None):
    """
    예측 수행 및 스마트 캐시 저장이 포함된 함수 (수정됨)
    """
    try:
        logger.info(f"Starting prediction with smart cache save for {current_date}")
        
        # 기존 generate_predictions 함수 실행
        results = generate_predictions(df, current_date, predict_window, features, target_col, file_path)
        
        # 스마트 캐시 저장 옵션이 활성화된 경우
        if save_to_csv:
            logger.info("Saving prediction with smart cache system...")
            
            # 새로운 스마트 캐시 저장 함수 사용
            save_result = save_prediction_simple(results, current_date)
            results['save_info'] = save_result
            
            if save_result['success']:
                logger.info(f"✅ Smart cache save completed successfully")
                logger.info(f"  - Prediction Start Date: {save_result.get('prediction_start_date')}")
                logger.info(f"  - File: {save_result.get('file', 'N/A')}")
                
                # 캐시 정보 추가 (안전한 키 접근)
                results['cache_info'] = {
                    'saved': True,
                    'prediction_start_date': save_result.get('prediction_start_date'),
                    'file': save_result.get('file'),
                    'success': save_result.get('success', False)
                }
            else:
                logger.warning(f"❌ Failed to save prediction with smart cache: {save_result.get('error')}")
                results['cache_info'] = {
                    'saved': False,
                    'error': save_result.get('error')
                }
        else:
            logger.info("Skipping smart cache save (save_to_csv=False)")
            results['save_info'] = {'success': False, 'reason': 'save_to_csv=False'}
            results['cache_info'] = {
                'saved': False,
                'reason': 'save_to_csv=False'
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in generate_predictions_with_save: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 오류 발생 시에도 예측 결과는 반환하되, 저장 실패 정보 포함
        if 'results' in locals():
            results['save_info'] = {'success': False, 'error': str(e)}
            results['cache_info'] = {'saved': False, 'error': str(e)}
            return results
        else:
            # 예측 자체가 실패한 경우
            raise e

def generate_predictions_with_attention_save(df, current_date, predict_window=23, features=None, target_col='MOPJ', save_to_csv=True, file_path=None):
    """
    예측 수행 및 attention 포함 CSV 저장 함수
    
    Parameters:
    -----------
    df : pandas.DataFrame
        전체 데이터
    current_date : str or datetime
        현재 날짜 (데이터 기준일)
    predict_window : int
        예측 기간 (기본 23일)
    features : list, optional
        사용할 특성 목록
    target_col : str
        타겟 컬럼명 (기본 'MOPJ')
    save_to_csv : bool
        CSV 저장 여부 (기본 True)
    
    Returns:
    --------
    dict : 예측 결과 (attention 데이터 포함)
    """
    try:
        logger.info(f"Starting prediction with attention save for {current_date}")
        
        # 기존 generate_predictions 함수 실행
        results = generate_predictions(df, current_date, predict_window, features, target_col, file_path)
        
        # attention 포함 저장 옵션이 활성화된 경우
        if save_to_csv:
            logger.info("Saving prediction with attention data...")
            save_result = save_prediction_simple(results, current_date)
            results['save_info'] = save_result
            
            if save_result['success']:
                logger.info(f"SUCCESS: Prediction with attention saved successfully")
                logger.info(f"  - CSV: {save_result['csv_file']}")
                logger.info(f"  - Metadata: {save_result['meta_file']}")
                logger.info(f"  - Attention: {save_result['attention_file'] if save_result.get('attention_file') else 'Not saved'}")
            else:
                logger.warning(f"❌ Failed to save prediction with attention: {save_result.get('error')}")
        else:
            logger.info("Skipping CSV save (save_to_csv=False)")
            results['save_info'] = {'success': False, 'reason': 'save_to_csv=False'}
        
        return results
        
    except Exception as e:
        logger.error(f"Error in generate_predictions_with_attention_save: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 오류 발생 시에도 예측 결과는 반환하되, 저장 실패 정보 포함
        if 'results' in locals():
            results['save_info'] = {'success': False, 'error': str(e)}
            return results
        else:
            # 예측 자체가 실패한 경우
            raise e

#######################################################################
# 백그라운드 작업 처리
#######################################################################
# 🔧 SyntaxError 수정 - check_existing_prediction 함수 (3987라인 근처)

def check_existing_prediction(current_date):
    """
    파일별 디렉토리 구조에서 저장된 예측을 확인하고 불러오는 함수
    🎯 현재 파일의 디렉토리에서 우선 검색
    """
    try:
        # 현재 날짜(데이터 기준일)에서 첫 번째 예측 날짜 계산
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        # 다음 영업일 찾기 (현재 날짜의 다음 영업일이 첫 번째 예측 날짜)
        next_date = current_date + pd.Timedelta(days=1)
        while next_date.weekday() >= 5 or is_holiday(next_date):
            next_date += pd.Timedelta(days=1)
        
        first_prediction_date = next_date
        date_str = first_prediction_date.strftime('%Y%m%d')
        
        logger.info(f"🔍 Checking cache for prediction starting: {first_prediction_date.strftime('%Y-%m-%d')}")
        logger.info(f"  📅 Data end date: {current_date.strftime('%Y-%m-%d')}")
        logger.info(f"  📅 Expected prediction start: {first_prediction_date.strftime('%Y-%m-%d')}")
        logger.info(f"  📄 Expected filename pattern: prediction_start_{date_str}.*")
        
        # 🎯 1단계: 현재 파일의 캐시 디렉토리에서 정확한 날짜 매치로 캐시 찾기
        try:
            cache_dirs = get_file_cache_dirs()
            file_predictions_dir = cache_dirs['predictions']
            
            logger.info(f"  📁 Cache directory: {cache_dirs['root']}")
            logger.info(f"  📁 Predictions directory: {file_predictions_dir}")
            logger.info(f"  📁 Directory exists: {file_predictions_dir.exists()}")
            
        except Exception as e:
            logger.error(f"❌ Failed to get cache directories: {str(e)}")
            return None
        
        if file_predictions_dir.exists():
            exact_csv = file_predictions_dir / f"prediction_start_{date_str}.csv"
            exact_meta = file_predictions_dir / f"prediction_start_{date_str}_meta.json"
            
            logger.info(f"  🔍 Looking for: {exact_csv}")
            logger.info(f"  🔍 CSV exists: {exact_csv.exists()}")
            logger.info(f"  🔍 Meta exists: {exact_meta.exists()}")
            
            if exact_csv.exists() and exact_meta.exists():
                logger.info(f"✅ Found exact prediction cache in file directory: {exact_csv.name}")
                return load_prediction_with_attention_from_csv_in_dir(first_prediction_date, file_predictions_dir)
            
            # 해당 파일 디렉토리에서 다른 날짜의 예측 찾기
            logger.info("🔍 Searching for other predictions in file directory...")
            prediction_files = list(file_predictions_dir.glob("prediction_start_*_meta.json"))
            
            logger.info(f"  📋 Found {len(prediction_files)} prediction files:")
            for i, pf in enumerate(prediction_files):
                logger.info(f"    {i+1}. {pf.name}")
            
            if prediction_files:
                # 가장 최근 예측 사용 (임시 방편)
                latest_file = max(prediction_files, key=lambda x: x.stem)
                cached_date_str = latest_file.stem.replace('prediction_start_', '').replace('_meta', '')
                cached_prediction_date = pd.to_datetime(cached_date_str, format='%Y%m%d')
                
                logger.info(f"🎯 Found compatible prediction in file directory!")
                logger.info(f"  📅 Cached prediction date: {cached_prediction_date.strftime('%Y-%m-%d')}")
                logger.info(f"  📄 Using file: {latest_file.name}")
                
                return load_prediction_with_attention_from_csv_in_dir(cached_prediction_date, file_predictions_dir)
        else:
            logger.warning(f"❌ Predictions directory does not exist: {file_predictions_dir}")
        
        # 🎯 2단계: 다른 파일 캐시 디렉토리에서 호환 캐시 찾기
        logger.info("🔍 Searching in other file cache directories...")
        
        cache_root = Path(CACHE_ROOT_DIR)
        if not cache_root.exists():
            logger.info("❌ Cache root directory does not exist")
            return None
        
        current_file_path = prediction_state.get('current_file', None)
        logger.info(f"  📂 Current file: {current_file_path}")
        
        # 모든 파일 캐시 디렉토리 스캔
        other_dirs_checked = 0
        for file_dir in cache_root.iterdir():
            if not file_dir.is_dir() or file_dir.name == "default":
                continue
            
            # 현재 파일의 디렉토리는 이미 확인했으므로 건너뛰기
            if file_dir == cache_dirs['root']:
                continue
            
            other_dirs_checked += 1
            logger.info(f"  🔍 Checking other directory: {file_dir.name}")
            
            # 각 파일 캐시 디렉토리의 predictions 하위 디렉토리에서 검색
            other_predictions_dir = file_dir / 'predictions'
            if not other_predictions_dir.exists():
                logger.info(f"    ❌ No predictions subdirectory in {file_dir.name}")
                continue
                
            prediction_files = list(other_predictions_dir.glob("prediction_start_*_meta.json"))
            logger.info(f"    📋 Found {len(prediction_files)} prediction files in {file_dir.name}")
            for meta_file in prediction_files:
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    
                    # 파일 해시 비교
                    current_file_hash = get_data_content_hash(current_file_path) if current_file_path else None
                    cached_file_hash = meta_data.get('file_content_hash')
                    
                    logger.info(f"    🔍 Checking {meta_file.name}:")
                    logger.info(f"      📝 Current file hash: {current_file_hash[:12] if current_file_hash else 'None'}...")
                    logger.info(f"      📝 Cached file hash:  {cached_file_hash[:12] if cached_file_hash else 'None'}...")
                    
                    if cached_file_hash and cached_file_hash == current_file_hash:
                        # 동일한 파일 내용에서 생성된 예측 발견
                        cached_date_str = meta_file.stem.replace('prediction_start_', '').replace('_meta', '')
                        cached_prediction_date = pd.to_datetime(cached_date_str, format='%Y%m%d')
                        
                        logger.info(f"🎯 Found compatible prediction cache in other directory!")
                        logger.info(f"  📁 Directory: {file_dir.name}")
                        logger.info(f"  📅 Cached prediction date: {cached_prediction_date.strftime('%Y-%m-%d')}")
                        logger.info(f"  📝 File hash match: {cached_file_hash[:12]}...")
                        
                        return load_prediction_with_attention_from_csv_in_dir(cached_prediction_date, other_predictions_dir)
                        
                except Exception as e:
                    logger.debug(f"  ⚠️  Error reading meta file {meta_file}: {str(e)}")
                    continue
        
        logger.info(f"🔍 Summary: Checked {other_dirs_checked} other cache directories")
        logger.info("❌ No compatible prediction cache found")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error checking existing prediction: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_visualizations_realtime(predictions, df, current_date, metadata):
    """실시간으로 시각화 생성 (저장하지 않음)"""
    try:
        # DataFrame으로 변환
        sequence_df = pd.DataFrame(predictions)
        if 'Date' in sequence_df.columns:
            sequence_df['Date'] = pd.to_datetime(sequence_df['Date'])
        
        # 시작값 계산
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        start_day_value = df.loc[current_date, 'MOPJ'] if current_date in df.index else None
        
        if start_day_value is not None:
            # 성능 메트릭 계산
            metrics = compute_performance_metrics_improved(sequence_df, start_day_value)
            
            # 기본 그래프 생성 (메모리에만)
            _, basic_plot_img = plot_prediction_basic(
                sequence_df, 
                metadata.get('prediction_start_date', current_date),
                start_day_value,
                metrics['f1'],
                metrics['accuracy'], 
                metrics['mape'],
                metrics['weighted_score'],
                save_prefix=None  # 파일별 캐시 디렉토리 자동 사용
            )
            
            # 이동평균 계산 및 시각화
            historical_data = df[df.index <= current_date].copy()
            ma_results = calculate_moving_averages_with_history(predictions, historical_data, target_col='MOPJ')
            
            _, ma_plot_img = plot_moving_average_analysis(
                ma_results,
                metadata.get('prediction_start_date', current_date),
                save_prefix=None  # 파일별 캐시 디렉토리 자동 사용
            )
            
            # 상태에 저장
            prediction_state['latest_plots'] = {
                'basic_plot': {'file': None, 'image': basic_plot_img},
                'ma_plot': {'file': None, 'image': ma_plot_img}
            }
            prediction_state['latest_ma_results'] = ma_results
            prediction_state['latest_metrics'] = metrics
            
        else:
            logger.warning("Cannot generate visualizations: start day value not available")
            prediction_state['latest_plots'] = {
                'basic_plot': {'file': None, 'image': None},
                'ma_plot': {'file': None, 'image': None}
            }
            prediction_state['latest_ma_results'] = {}
            prediction_state['latest_metrics'] = {}
            
    except Exception as e:
        logger.error(f"Error generating realtime visualizations: {str(e)}")
        prediction_state['latest_plots'] = {
            'basic_plot': {'file': None, 'image': None},
            'ma_plot': {'file': None, 'image': None}
        }
        prediction_state['latest_ma_results'] = {}
        prediction_state['latest_metrics'] = {}

def regenerate_visualizations_from_cache(predictions, df, current_date, metadata):
    """
    캐시된 데이터로부터 시각화를 재생성하는 함수
    🔑 current_date를 전달하여 과거/미래 구분 시각화 생성
    """
    try:
        logger.info("🎨 Regenerating visualizations from cached data...")
        
        # DataFrame으로 변환 (안전한 방식)
        temp_df_for_plot = pd.DataFrame([
            {
                'Date': pd.to_datetime(item.get('Date') or item.get('date')),
                'Prediction': safe_serialize_value(item.get('Prediction') or item.get('prediction')),
                'Actual': safe_serialize_value(item.get('Actual') or item.get('actual'))
            } for item in predictions if item.get('Date') or item.get('date')
        ])
        
        logger.info(f"  📊 Plot data prepared: {len(temp_df_for_plot)} predictions")
        
        # current_date 처리
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        # 시작값 계산
        start_day_value = None
        if current_date in df.index and not pd.isna(df.loc[current_date, 'MOPJ']):
            start_day_value = df.loc[current_date, 'MOPJ']
            logger.info(f"  📈 Start day value: {start_day_value:.2f}")
        else:
            logger.warning(f"  ⚠️  Start day value not available for {current_date}")
        
        # 메타데이터에서 메트릭 가져오기 (안전한 방식)
        metrics = metadata.get('metrics')
        if metrics:
            f1_score = safe_serialize_value(metrics.get('f1', 0.0))
            accuracy = safe_serialize_value(metrics.get('accuracy', 0.0))
            mape = safe_serialize_value(metrics.get('mape', 0.0))
            weighted_score = safe_serialize_value(metrics.get('weighted_score', 0.0))
            logger.info(f"  📊 Metrics loaded - F1: {f1_score:.3f}, Acc: {accuracy:.1f}%, MAPE: {mape:.1f}%")
        else:
            f1_score = accuracy = mape = weighted_score = 0.0
            logger.info("  ℹ️  No metrics available - using default values")
        
        plots = {
            'basic_plot': {'file': None, 'image': None},
            'ma_plot': {'file': None, 'image': None}
        }
        
        # 시각화 생성 (데이터가 충분한 경우만)
        if start_day_value is not None and not temp_df_for_plot.empty:
            logger.info("  🎨 Generating basic prediction plot...")
            
            # 예측 시작일 계산
            prediction_start_date = metadata.get('prediction_start_date')
            if isinstance(prediction_start_date, str):
                prediction_start_date = pd.to_datetime(prediction_start_date)
            elif prediction_start_date is None:
                # 메타데이터에 없으면 current_date 다음 영업일로 계산
                prediction_start_date = current_date + pd.Timedelta(days=1)
                while prediction_start_date.weekday() >= 5 or is_holiday(prediction_start_date):
                    prediction_start_date += pd.Timedelta(days=1)
                logger.info(f"  📅 Calculated prediction start date: {prediction_start_date}")
            
            # ✅ 핵심 수정: current_date 전달하여 과거/미래 구분 시각화
            basic_plot_file, basic_plot_img = plot_prediction_basic(
                temp_df_for_plot,
                prediction_start_date,
                start_day_value,
                f1_score,
                accuracy,
                mape,
                weighted_score,
                current_date=current_date,  # 🔑 핵심 수정: current_date 전달
                save_prefix=None,  # 파일별 캐시 디렉토리 자동 사용
                title_prefix="Cached Prediction Analysis"
            )
            
            if basic_plot_file:
                logger.info(f"  ✅ Basic plot generated: {basic_plot_file}")
            else:
                logger.warning("  ❌ Basic plot generation failed")
            
            # 이동평균 계산 및 시각화
            logger.info("  📈 Calculating moving averages...")
            historical_data = df[df.index <= current_date].copy()
            
            # 캐시된 예측 데이터를 이동평균 계산용으로 변환
            ma_input_data = []
            for pred in predictions:
                try:
                    ma_item = {
                        'Date': pd.to_datetime(pred.get('Date') or pred.get('date')),
                        'Prediction': safe_serialize_value(pred.get('Prediction') or pred.get('prediction')),
                        'Actual': safe_serialize_value(pred.get('Actual') or pred.get('actual'))
                    }
                    ma_input_data.append(ma_item)
                except Exception as e:
                    logger.warning(f"  ⚠️  Error processing MA data item: {str(e)}")
                    continue
            
            ma_results = calculate_moving_averages_with_history(
                ma_input_data, historical_data, target_col='MOPJ'
            )
            
            if ma_results:
                logger.info(f"  📊 MA calculated for {len(ma_results)} windows")
                
                # 이동평균 시각화
                ma_plot_file, ma_plot_img = plot_moving_average_analysis(
                    ma_results,
                    prediction_start_date,
                    save_prefix=None,  # 파일별 캐시 디렉토리 자동 사용
                    title_prefix="Cached Moving Average Analysis"
                )
                
                if ma_plot_file:
                    logger.info(f"  ✅ MA plot generated: {ma_plot_file}")
                else:
                    logger.warning("  ❌ MA plot generation failed")
            else:
                logger.warning("  ⚠️  Moving average calculation failed")
                ma_plot_file, ma_plot_img = None, None
            
            plots = {
                'basic_plot': {'file': basic_plot_file, 'image': basic_plot_img},
                'ma_plot': {'file': ma_plot_file, 'image': ma_plot_img}
            }
            
            logger.info("  ✅ Visualizations regenerated from cache successfully")
        else:
            if start_day_value is None:
                logger.warning("  ❌ Cannot regenerate visualizations: start day value not available")
            if temp_df_for_plot.empty:
                logger.warning("  ❌ Cannot regenerate visualizations: no prediction data")
        
        return plots
        
    except Exception as e:
        logger.error(f"❌ Error regenerating visualizations from cache: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            'basic_plot': {'file': None, 'image': None},
            'ma_plot': {'file': None, 'image': None}
        }

def background_prediction_simple_compatible(file_path, current_date, save_to_csv=True, use_cache=True):
    """호환성을 유지하는 백그라운드 예측 함수 - 캐시 우선 사용, JSON 안전성 보장"""
    global prediction_state
    
    try:
        prediction_state['is_predicting'] = True
        prediction_state['prediction_progress'] = 10
        prediction_state['error'] = None
        prediction_state['latest_file_path'] = file_path  # 파일 경로 저장
        prediction_state['current_file'] = file_path  # 캐시 연동용 파일 경로
        
        logger.info(f"🎯 Starting compatible prediction for {current_date}")
        logger.info(f"  🔄 Cache enabled: {use_cache}")
        
        # 데이터 로드
        df = load_data(file_path)
        prediction_state['current_data'] = df
        prediction_state['prediction_progress'] = 20
        
        # 현재 날짜 처리 및 영업일 조정
        if current_date is None:
            current_date = df.index.max()
        else:
            current_date = pd.to_datetime(current_date)
        
        # 🎯 휴일이면 다음 영업일로 조정
        original_date = current_date
        adjusted_date = current_date
        
        # 주말이나 휴일이면 다음 영업일로 이동
        while adjusted_date.weekday() >= 5 or is_holiday(adjusted_date):
            adjusted_date += pd.Timedelta(days=1)
        
        if adjusted_date != original_date:
            logger.info(f"📅 Date adjusted for business day: {original_date.strftime('%Y-%m-%d')} -> {adjusted_date.strftime('%Y-%m-%d')}")
            logger.info(f"  📋 Reason: {'Weekend' if original_date.weekday() >= 5 else 'Holiday'}")
        
        current_date = adjusted_date
        
        # 캐시 확인 (우선 사용)
        if use_cache:
            logger.info("🔍 Checking for existing prediction cache...")
            prediction_state['prediction_progress'] = 30
            
            try:
                cached_result = check_existing_prediction(current_date)
                logger.info(f"  📋 Cache check result: {cached_result is not None}")
                if cached_result:
                    logger.info(f"  📋 Cache success status: {cached_result.get('success', False)}")
                else:
                    logger.info("  ❌ No cache result returned")
            except Exception as cache_check_error:
                logger.error(f"  ❌ Cache check failed with error: {str(cache_check_error)}")
                logger.error(f"  📝 Error traceback: {traceback.format_exc()}")
                cached_result = None
            
            if cached_result and cached_result.get('success'):
                logger.info("🎉 Found existing prediction! Loading from cache...")
                prediction_state['prediction_progress'] = 50
                
                try:
                    # 캐시된 데이터 로드 및 정리
                    predictions = cached_result['predictions']
                    metadata = cached_result['metadata']
                    attention_data = cached_result.get('attention_data')
                    
                    # 데이터 정리 (JSON 안전성 보장)
                    cleaned_predictions = clean_cached_predictions(predictions)
                    
                    # 호환성 유지된 형태로 변환
                    compatible_predictions = convert_to_legacy_format(cleaned_predictions)
                    
                    # JSON 직렬화 테스트
                    try:
                        test_json = json.dumps(compatible_predictions[:1] if compatible_predictions else [])
                        logger.info("✅ JSON serialization test passed for cached data")
                    except Exception as json_error:
                        logger.error(f"❌ JSON serialization failed for cached data: {str(json_error)}")
                        raise Exception("Cached data serialization failed")
                    
                    # 구간 점수 처리 (JSON 안전)
                    interval_scores = metadata.get('interval_scores', {})
                    cleaned_interval_scores = {}
                    for key, value in interval_scores.items():
                        if isinstance(value, dict):
                            cleaned_score = {}
                            for k, v in value.items():
                                cleaned_score[k] = safe_serialize_value(v)
                            cleaned_interval_scores[key] = cleaned_score
                        else:
                            cleaned_interval_scores[key] = safe_serialize_value(value)
                    
                    # 이동평균 재계산
                    prediction_state['prediction_progress'] = 60
                    logger.info("Recalculating moving averages from cached data...")
                    historical_data = df[df.index <= current_date].copy()
                    ma_results = calculate_moving_averages_with_history(
                        cleaned_predictions, historical_data, target_col='MOPJ'
                    )
                    
                    # 시각화 재생성
                    prediction_state['prediction_progress'] = 70
                    logger.info("Regenerating visualizations from cached data...")
                    plots = regenerate_visualizations_from_cache(
                        cleaned_predictions, df, current_date, metadata
                    )
                    
                    # 메트릭 정리
                    metrics = metadata.get('metrics')
                    cleaned_metrics = {}
                    if metrics:
                        for key, value in metrics.items():
                            cleaned_metrics[key] = safe_serialize_value(value)
                    
                    # 어텐션 데이터 정리
                    cleaned_attention = None
                    if attention_data:
                        cleaned_attention = {}
                        for key, value in attention_data.items():
                            if key == 'image' and value:
                                cleaned_attention[key] = value  # base64 이미지는 그대로
                            elif isinstance(value, dict):
                                cleaned_attention[key] = {}
                                for k, v in value.items():
                                    cleaned_attention[key][k] = safe_serialize_value(v)
                            else:
                                cleaned_attention[key] = safe_serialize_value(value)
                    
                    # 상태 설정
                    prediction_state['latest_predictions'] = compatible_predictions
                    prediction_state['latest_attention_data'] = cleaned_attention
                    prediction_state['current_date'] = safe_serialize_value(metadata.get('prediction_start_date'))
                    prediction_state['selected_features'] = metadata.get('selected_features', [])
                    prediction_state['semimonthly_period'] = safe_serialize_value(metadata.get('semimonthly_period'))
                    prediction_state['next_semimonthly_period'] = safe_serialize_value(metadata.get('next_semimonthly_period'))
                    prediction_state['latest_interval_scores'] = cleaned_interval_scores
                    prediction_state['latest_metrics'] = cleaned_metrics
                    prediction_state['latest_plots'] = plots
                    prediction_state['latest_ma_results'] = ma_results
                    
                    # feature_importance 설정
                    if cleaned_attention and 'feature_importance' in cleaned_attention:
                        prediction_state['feature_importance'] = cleaned_attention['feature_importance']
                    else:
                        prediction_state['feature_importance'] = None
                    
                    prediction_state['prediction_progress'] = 100
                    prediction_state['is_predicting'] = False
                    logger.info("✅ Cache prediction completed successfully!")
                    return
                    
                except Exception as cache_error:
                    logger.warning(f"⚠️  Cache processing failed: {str(cache_error)}")
                    logger.info("🔄 Falling back to new prediction...")
            else:
                logger.info("  📋 No usable cache found - proceeding with new prediction")
        else:
            logger.info("🆕 Cache disabled - running new prediction...")
        
        # 새로운 예측 수행
        logger.info("🤖 Running new prediction...")
        prediction_state['prediction_progress'] = 40
        
        results = generate_predictions_compatible(df, current_date)
        prediction_state['prediction_progress'] = 80
        
        # 새로운 예측 결과 정리 (JSON 안전성 보장)
        if isinstance(results.get('predictions'), list):
            raw_predictions = results['predictions']
        else:
            raw_predictions = results.get('predictions_flat', [])
        
        # 호환성 유지된 형태로 변환
        compatible_predictions = convert_to_legacy_format(raw_predictions)
        
        # JSON 직렬화 테스트
        try:
            test_json = json.dumps(compatible_predictions[:1] if compatible_predictions else [])
            logger.info("✅ JSON serialization test passed for new prediction")
        except Exception as json_error:
            logger.error(f"❌ JSON serialization failed for new prediction: {str(json_error)}")
            # 데이터 추가 정리 시도
            for pred in compatible_predictions:
                for key, value in pred.items():
                    pred[key] = safe_serialize_value(value)
        
        # 상태 설정
        prediction_state['latest_predictions'] = compatible_predictions
        prediction_state['latest_attention_data'] = results.get('attention_data')
        prediction_state['current_date'] = safe_serialize_value(results.get('current_date'))
        prediction_state['selected_features'] = results.get('selected_features', [])
        prediction_state['semimonthly_period'] = safe_serialize_value(results.get('semimonthly_period'))
        prediction_state['next_semimonthly_period'] = safe_serialize_value(results.get('next_semimonthly_period'))
        prediction_state['latest_interval_scores'] = results.get('interval_scores', {})
        prediction_state['latest_metrics'] = results.get('metrics')
        prediction_state['latest_plots'] = results.get('plots', {})
        prediction_state['latest_ma_results'] = results.get('ma_results', {})
        
        # feature_importance 설정
        if results.get('attention_data') and 'feature_importance' in results['attention_data']:
            prediction_state['feature_importance'] = results['attention_data']['feature_importance']
        else:
            prediction_state['feature_importance'] = None
        
        # 저장
        if save_to_csv:
            logger.info("💾 Saving prediction to cache...")
            save_result = save_prediction_simple(results, current_date)
            if save_result['success']:
                logger.info(f"✅ Cache saved successfully: {save_result.get('prediction_start_date')}")
            else:
                logger.warning(f"⚠️  Cache save failed: {save_result.get('error')}")
        
        prediction_state['prediction_progress'] = 100
        prediction_state['is_predicting'] = False
        logger.info("✅ New prediction completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error in compatible prediction: {str(e)}")
        logger.error(traceback.format_exc())
        prediction_state['error'] = str(e)
        prediction_state['is_predicting'] = False
        prediction_state['prediction_progress'] = 0


def safe_serialize_value(value):
    """값을 JSON 안전하게 직렬화 (배열 타입 처리 개선)"""
    if value is None:
        return None
    
    # numpy/pandas 배열 타입 먼저 체크
    if isinstance(value, (np.ndarray, pd.Series, list)):
        if len(value) == 0:
            return []
        elif len(value) == 1:
            # 단일 원소 배열인 경우 스칼라로 처리
            return safe_serialize_value(value[0])
        else:
            # 다중 원소 배열인 경우 리스트로 변환
            try:
                return [safe_serialize_value(item) for item in value]
            except:
                return [str(item) for item in value]
    
    # 스칼라 값에 대해서만 pd.isna 체크
    try:
        if pd.isna(value):  # 스칼라 값에 대해서만 사용
            return None
    except (TypeError, ValueError):
        # pd.isna가 처리할 수 없는 타입인 경우 넘어감
        pass
    
    if isinstance(value, (int, float)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    elif isinstance(value, np.floating):  # numpy float 타입 처리
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    elif isinstance(value, np.integer):  # numpy int 타입 처리
        return int(value)
    elif isinstance(value, str):
        return value
    elif hasattr(value, 'isoformat'):  # datetime/Timestamp
        return value.strftime('%Y-%m-%d')
    elif hasattr(value, 'strftime'):  # 기타 날짜 객체
        return value.strftime('%Y-%m-%d')
    else:
        try:
            # JSON 직렬화 테스트
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

def clean_predictions_data(predictions):
    """예측 데이터를 JSON 안전하게 정리"""
    if not predictions:
        return []
    
    cleaned = []
    for pred in predictions:
        cleaned_pred = {}
        for key, value in pred.items():
            if key in ['date', 'prediction_from']:
                # 날짜 필드
                if hasattr(value, 'strftime'):
                    cleaned_pred[key] = value.strftime('%Y-%m-%d')
                else:
                    cleaned_pred[key] = str(value)
            elif key in ['prediction', 'actual', 'error', 'error_pct']:
                # 숫자 필드
                cleaned_pred[key] = safe_serialize_value(value)
            else:
                # 기타 필드
                cleaned_pred[key] = safe_serialize_value(value)
        cleaned.append(cleaned_pred)
    
    return cleaned

def clean_cached_predictions(predictions):
    """캐시에서 로드된 예측 데이터를 정리하는 함수"""
    cleaned_predictions = []
    
    for pred in predictions:
        try:
            # 모든 필드를 안전하게 처리
            cleaned_pred = {}
            for key, value in pred.items():
                if key in ['Date', 'date']:
                    # 날짜 필드 특별 처리
                    if pd.notna(value):
                        if hasattr(value, 'strftime'):
                            cleaned_pred[key] = value.strftime('%Y-%m-%d')
                        else:
                            cleaned_pred[key] = str(value)[:10]
                    else:
                        cleaned_pred[key] = None
                elif key in ['Prediction', 'prediction', 'Actual', 'actual']:
                    # 숫자 필드 처리
                    cleaned_pred[key] = safe_serialize_value(value)
                else:
                    # 기타 필드
                    cleaned_pred[key] = safe_serialize_value(value)
            
            cleaned_predictions.append(cleaned_pred)
            
        except Exception as e:
            logger.warning(f"Error cleaning prediction item: {str(e)}")
            continue
    
    return cleaned_predictions

def clean_interval_scores_safe(interval_scores):
    """구간 점수를 안전하게 정리하는 함수"""
    cleaned_interval_scores = []
    
    try:
        if isinstance(interval_scores, dict):
            for key, value in interval_scores.items():
                if isinstance(value, dict):
                    cleaned_score = {}
                    for k, v in value.items():
                        # 배열이나 복잡한 타입은 특별 처리
                        if isinstance(v, (np.ndarray, pd.Series, list)):
                            if len(v) == 1:
                                cleaned_score[k] = safe_serialize_value(v[0])
                            elif len(v) == 0:
                                cleaned_score[k] = None
                            else:
                                # 다중 원소 배열은 문자열로 변환
                                cleaned_score[k] = str(v)
                        else:
                            cleaned_score[k] = safe_serialize_value(v)
                    cleaned_interval_scores.append(cleaned_score)
                else:
                    # dict가 아닌 경우 안전하게 처리
                    cleaned_interval_scores.append(safe_serialize_value(value))
        elif isinstance(interval_scores, list):
            for score in interval_scores:
                if isinstance(score, dict):
                    cleaned_score = {}
                    for k, v in score.items():
                        # 배열이나 복잡한 타입은 특별 처리
                        if isinstance(v, (np.ndarray, pd.Series, list)):
                            if len(v) == 1:
                                cleaned_score[k] = safe_serialize_value(v[0])
                            elif len(v) == 0:
                                cleaned_score[k] = None
                            else:
                                cleaned_score[k] = str(v)
                        else:
                            cleaned_score[k] = safe_serialize_value(v)
                    cleaned_interval_scores.append(cleaned_score)
                else:
                    cleaned_interval_scores.append(safe_serialize_value(score))
        
        return cleaned_interval_scores
        
    except Exception as e:
        logger.error(f"Error cleaning interval scores: {str(e)}")
        return []

def convert_to_legacy_format(predictions_data):
    """
    새·옛 구조를 모두 받아 프론트엔드(대문자) + 백엔드(소문자) 키를 동시 보존.
    JSON 직렬화 안전성 보장
    """
    if not predictions_data:
        return []
    
    legacy_out = []
    for pred in predictions_data:
        try:
            # 날짜 필드 안전 처리
            date_value = pred.get("date") or pred.get("Date")
            if hasattr(date_value, 'strftime'):
                date_str = date_value.strftime('%Y-%m-%d')
            elif isinstance(date_value, str):
                date_str = date_value[:10] if len(date_value) > 10 else date_value
            else:
                date_str = str(date_value) if date_value is not None else None
            
            # 예측값 안전 처리
            prediction_value = pred.get("prediction") or pred.get("Prediction")
            prediction_safe = safe_serialize_value(prediction_value)
            
            # 실제값 안전 처리
            actual_value = pred.get("actual") or pred.get("Actual")
            actual_safe = safe_serialize_value(actual_value)
            
            # 기타 필드들 안전 처리
            prediction_from = pred.get("prediction_from")
            if hasattr(prediction_from, 'strftime'):
                prediction_from = prediction_from.strftime('%Y-%m-%d')
            elif prediction_from:
                prediction_from = str(prediction_from)
            
            legacy_item = {
                # ── 프론트엔드 호환 대문자 키 (JSON 안전) ───────────────
                "Date": date_str,
                "Prediction": prediction_safe,
                "Actual": actual_safe,

                # ── 백엔드 후속 함수(소문자 'date' 참조)용 ──
                "date": date_str,
                "prediction": prediction_safe,
                "actual": actual_safe,

                # 기타 필드 안전 처리
                "Prediction_From": prediction_from,
                "SemimonthlyPeriod": safe_serialize_value(pred.get("semimonthly_period")),
                "NextSemimonthlyPeriod": safe_serialize_value(pred.get("next_semimonthly_period")),
                "is_synthetic": bool(pred.get("is_synthetic", False)),
                
                # 추가 메타데이터 (있는 경우)
                "day_offset": safe_serialize_value(pred.get("day_offset")),
                "is_business_day": bool(pred.get("is_business_day", True)),
                "error": safe_serialize_value(pred.get("error")),
                "error_pct": safe_serialize_value(pred.get("error_pct"))
            }
            
            legacy_out.append(legacy_item)
            
        except Exception as e:
            logger.warning(f"Error converting prediction item: {str(e)}")
            continue
    
    return legacy_out

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

@app.route('/api/test/cache-dirs', methods=['GET'])
def test_cache_dirs():
    """캐시 디렉토리 시스템 테스트"""
    try:
        # 현재 상태 확인
        current_file = prediction_state.get('current_file', None)
        
        # 파일 경로가 있으면 해당 파일로, 없으면 기본으로 테스트
        test_file = request.args.get('file_path', current_file)
        
        if test_file and not os.path.exists(test_file):
            return jsonify({
                'error': f'File does not exist: {test_file}',
                'current_file': current_file
            }), 400
        
        # 캐시 디렉토리 생성 테스트
        cache_dirs = get_file_cache_dirs(test_file)
        
        # 디렉토리 존재 여부 확인
        dir_status = {}
        for name, path in cache_dirs.items():
            dir_status[name] = {
                'path': str(path),
                'exists': path.exists(),
                'is_dir': path.is_dir() if path.exists() else False
            }
        
        return jsonify({
            'success': True,
            'test_file': test_file,
            'current_file': current_file,
            'cache_dirs': dir_status,
            'cache_root_exists': Path(CACHE_ROOT_DIR).exists()
        })
        
    except Exception as e:
        logger.error(f"Cache directory test failed: {str(e)}")
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """스마트 캐시 기능이 있는 CSV 파일 업로드 API"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and file.filename.endswith('.csv'):
        try:
            # 임시 파일명 생성
            original_filename = secure_filename(file.filename)
            temp_filename = secure_filename(f"temp_{int(time.time())}.csv")
            temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
            
            # 임시 파일로 저장
            file.save(temp_filepath)
            logger.info(f"📤 [UPLOAD] File saved temporarily: {temp_filename}")
            
            # 🔍 캐시 호환성 확인
            cache_result = find_compatible_cache_file(temp_filepath)
            
            response_data = {
                'success': True,
                'filepath': temp_filepath,
                'filename': temp_filename,
                'original_filename': original_filename,
                'cache_info': {
                    'found': cache_result['found'],
                    'cache_type': cache_result.get('cache_type'),
                    'message': None
                }
            }
            
            if cache_result['found']:
                cache_type = cache_result['cache_type']
                cache_file = cache_result['cache_file']
                
                if cache_type == 'exact':
                    response_data['cache_info']['message'] = f"동일한 데이터 발견! 기존 캐시를 활용합니다. ({os.path.basename(cache_file)})"
                    response_data['cache_info']['compatible_file'] = cache_file
                    logger.info(f"✅ [CACHE] Exact match found: {cache_file}")
                    
                elif cache_type == 'extension':
                    ext_info = cache_result['extension_info']
                    response_data['cache_info']['message'] = f"데이터 확장 감지! {ext_info['new_rows_count']}개 새 행이 추가되었습니다. 기존 캐시를 활용합니다."
                    response_data['cache_info']['compatible_file'] = cache_file
                    response_data['cache_info']['extension_info'] = ext_info
                    logger.info(f"📈 [CACHE] Extension detected: +{ext_info['new_rows_count']} rows from {cache_file}")
                    
                # 호환 파일을 실제 파일 경로로 설정 (캐시 활용을 위해)
                response_data['filepath'] = cache_file
                response_data['filename'] = os.path.basename(cache_file)
                
                # 임시 파일 삭제 (필요시)
                if temp_filepath != cache_file:
                    try:
                        os.remove(temp_filepath)
                        logger.info(f"🗑️ [CLEANUP] Temporary file removed: {temp_filename}")
                    except:
                        pass
            else:
                # 새 파일인 경우 정식 파일명으로 변경
                content_hash = get_data_content_hash(temp_filepath)
                final_filename = f"data_{content_hash}.csv" if content_hash else temp_filename
                final_filepath = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
                
                if temp_filepath != final_filepath:
                    shutil.move(temp_filepath, final_filepath)
                    logger.info(f"📝 [UPLOAD] File renamed: {final_filename}")
                    
                response_data['filepath'] = final_filepath
                response_data['filename'] = final_filename
                response_data['cache_info']['message'] = "새로운 데이터입니다. 예측 후 캐시로 저장됩니다."
            
            # 🔑 업로드된 파일 경로를 전역 상태에 저장
            prediction_state['current_file'] = response_data['filepath']
            logger.info(f"📁 Set current_file in prediction_state: {response_data['filepath']}")
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error during file upload: {str(e)}")
            # 임시 파일 정리
            try:
                if 'temp_filepath' in locals() and os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
            except:
                pass
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Only CSV files are allowed'}), 400

@app.route('/api/holidays', methods=['GET'])
def get_holidays():
    """휴일 목록 조회 API"""
    try:
        # 휴일을 날짜와 설명이 포함된 딕셔너리 리스트로 변환
        holidays_list = []
        file_holidays = load_holidays_from_file()  # 파일에서 로드
        
        # 현재 전역 휴일에서 파일 휴일과 자동 감지 휴일 구분
        auto_detected = holidays - file_holidays
        
        for holiday_date in file_holidays:
            holidays_list.append({
                'date': holiday_date,
                'description': 'Holiday (from file)',
                'source': 'file'
            })
        
        for holiday_date in auto_detected:
            holidays_list.append({
                'date': holiday_date,
                'description': 'Holiday (detected from missing data)',
                'source': 'auto_detected'
            })
        
        # 날짜순으로 정렬
        holidays_list.sort(key=lambda x: x['date'])
        
        return jsonify({
            'success': True,
            'holidays': holidays_list,
            'count': len(holidays_list),
            'file_holidays': len(file_holidays),
            'auto_detected_holidays': len(auto_detected)
        })
    except Exception as e:
        logger.error(f"Error getting holidays: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'holidays': [],
            'count': 0
        }), 500

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
    days_limit = int(request.args.get('limit', 999999))  # 기본값을 매우 큰 수로 설정 (모든 날짜)
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # 🏖️ 데이터를 로드한 후 휴일 정보 자동 업데이트 (빈 평일 감지) - 임시 비활성화
        logger.info(f"🏖️ [HOLIDAYS] Auto-detection temporarily disabled to show more dates...")
        # updated_holidays = update_holidays(df=df)
        updated_holidays = load_holidays_from_file()  # 파일 휴일만 사용
        logger.info(f"🏖️ [HOLIDAYS] Total holidays (file only): {len(updated_holidays)}")
        
        # 전체 데이터의 50% 지점 계산 (참고용, 실제 필터링에는 사용하지 않음)
        total_rows = len(df)
        halfway_index = total_rows // 2
        halfway_date = df.iloc[halfway_index]['Date']
        
        logger.info(f"📊 Total data rows: {total_rows}")
        logger.info(f"📍 50% point: row {halfway_index}, date: {halfway_date.strftime('%Y-%m-%d')}")
        
        # 50% 지점에서 다음 반월 시작일 계산 (참고용)
        halfway_semimonthly = get_semimonthly_period(halfway_date)
        next_semimonthly = get_next_semimonthly_period(halfway_date)
        prediction_start_threshold, _ = get_semimonthly_date_range(next_semimonthly)
        
        logger.info(f"📅 50% point semimonthly period: {halfway_semimonthly}")
        logger.info(f"🎯 Next semimonthly period: {next_semimonthly}")
        logger.info(f"🚀 Prediction start threshold: {prediction_start_threshold.strftime('%Y-%m-%d')}")
        
        # 🔧 50% 지점 이후만 예측 가능한 날짜로 설정
        predictable_dates = df.iloc[halfway_index:]['Date']
        
        # 예측 가능한 모든 날짜를 내림차순으로 반환 (최신 날짜부터)
        # days_limit보다 작은 경우에만 제한 적용
        if len(predictable_dates) <= days_limit:
            dates = predictable_dates.sort_values(ascending=False).dt.strftime('%Y-%m-%d').tolist()
        else:
            dates = predictable_dates.sort_values(ascending=False).head(days_limit).dt.strftime('%Y-%m-%d').tolist()
        
        logger.info(f"🔢 Predictable dates count: {len(predictable_dates)} → 반환: {len(dates)}개")
        
        response_data = {
            'success': True,
            'dates': dates,
            'latest_date': dates[0] if dates else None,  # 첫 번째 요소가 최신 날짜 (내림차순)
            'prediction_threshold': prediction_start_threshold.strftime('%Y-%m-%d'),
            'halfway_point': halfway_date.strftime('%Y-%m-%d'),
            'halfway_semimonthly': halfway_semimonthly,
            'target_semimonthly': next_semimonthly
        }
        
        logger.info(f"📡 [API RESPONSE] Sending dates response:")
        logger.info(f"  📅 Total predictable dates: {len(predictable_dates)}")
        logger.info(f"  📅 Returned dates: {len(dates)}")
        logger.info(f"  📍 50% threshold: {response_data['prediction_threshold']}")
        logger.info(f"  🎯 Target period: {response_data['target_semimonthly']}")
        logger.info(f"  📅 Date range: {dates[-1]} ~ {dates[0]} (최신부터)")  # 첫번째가 최신, 마지막이 가장 오래된
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error reading dates: {str(e)}")
        return jsonify({'error': f'Error reading dates: {str(e)}'}), 500

@app.route('/api/predictions/saved', methods=['GET'])
def get_saved_predictions():
    """저장된 예측 결과 목록 조회 API"""
    try:
        limit = int(request.args.get('limit', 100))
        predictions_list = get_saved_predictions_list(limit)
        
        return jsonify({
            'success': True,
            'predictions': predictions_list,
            'count': len(predictions_list)
        })
    except Exception as e:
        logger.error(f"Error getting saved predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/saved/<date>', methods=['GET'])
def get_saved_prediction_by_date(date):
    """특정 날짜의 저장된 예측 결과 조회 API"""
    try:
        result = load_prediction_from_csv(date)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({'error': result['error']}), 404
    except Exception as e:
        logger.error(f"Error loading saved prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/saved/<date>', methods=['DELETE'])
def delete_saved_prediction_api(date):
    """저장된 예측 결과 삭제 API"""
    try:
        result = delete_saved_prediction(date)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify({'error': result['error']}), 500
    except Exception as e:
        logger.error(f"Error deleting saved prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/export', methods=['GET'])
def export_predictions():
    """저장된 예측 결과들을 하나의 CSV 파일로 내보내기 API"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        # 날짜 범위에 따른 예측 로드
        if start_date:
            predictions = load_accumulated_predictions_from_csv(start_date, end_date)
        else:
            # 모든 저장된 예측 로드
            predictions_list = get_saved_predictions_list(limit=1000)
            predictions = []
            for pred_info in predictions_list:
                loaded = load_prediction_from_csv(pred_info['prediction_date'])
                if loaded['success']:
                    predictions.extend(loaded['predictions'])
        
        if not predictions:
            return jsonify({'error': 'No predictions found for export'}), 404
        
        # DataFrame으로 변환
        if isinstance(predictions[0], dict) and 'predictions' in predictions[0]:
            # 누적 예측 형식인 경우
            all_predictions = []
            for pred_group in predictions:
                all_predictions.extend(pred_group['predictions'])
            export_df = pd.DataFrame(all_predictions)
        else:
            # 단순 예측 리스트인 경우
            export_df = pd.DataFrame(predictions)
        
        # 임시 파일 생성
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        export_df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        # 파일 전송
        return send_file(
            temp_file.name,
            as_attachment=True,
            download_name=f'predictions_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mimetype='text/csv'
        )
        
    except Exception as e:
        logger.error(f"Error exporting predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 7. API 엔드포인트 수정 - 스마트 캐시 사용
@app.route('/api/predict', methods=['POST'])
def start_prediction_compatible():
    """호환성을 유지하는 예측 시작 API - 캐시 우선 사용 (로그 강화)"""
    global prediction_state
    
    if prediction_state['is_predicting']:
        return jsonify({
            'success': False,
            'error': 'Prediction already in progress',
            'progress': prediction_state['prediction_progress']
        }), 409
    
    data = request.json
    filepath = data.get('filepath')
    current_date = data.get('date')
    save_to_csv = data.get('save_to_csv', True)
    use_cache = data.get('use_cache', True)  # 기본값 True
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400
    
    # 🔑 파일 경로를 전역 상태에 저장 (캐시 연동용)
    prediction_state['current_file'] = filepath
    
    # ✅ 로그 강화
    logger.info(f"🚀 Prediction API called:")
    logger.info(f"  📅 Target date: {current_date}")
    logger.info(f"  📁 Data file: {filepath}")
    logger.info(f"  💾 Save to CSV: {save_to_csv}")
    logger.info(f"  🔄 Use cache: {use_cache}")
    
    # 호환성 유지 백그라운드 함수 실행 (캐시 우선 사용)
    thread = Thread(target=background_prediction_simple_compatible, 
                   args=(filepath, current_date, save_to_csv, use_cache))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Compatible prediction started (cache-first)',
        'use_cache': use_cache,
        'cache_priority': 'high',
        'features': ['Cache-first loading', 'Unified file naming', 'Enhanced logging', 'Past/Future visualization split']
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
def get_prediction_results_compatible():
    """호환성을 유지하는 예측 결과 조회 API (오류 수정)"""
    global prediction_state
    
    logger.info(f"=== API /results called (compatible version) ===")
    
    if prediction_state['is_predicting']:
        return jsonify({
            'success': False,
            'error': 'Prediction in progress',
            'progress': prediction_state['prediction_progress']
        }), 409
    
    if prediction_state['latest_predictions'] is None:
        return jsonify({'error': 'No prediction results available'}), 404

    try:
        # 예측 데이터를 기존 형태로 변환
        if isinstance(prediction_state['latest_predictions'], list):
            raw_predictions = prediction_state['latest_predictions']
        else:
            raw_predictions = prediction_state['latest_predictions']
        
        # 기존 형태로 변환
        compatible_predictions = convert_to_legacy_format(raw_predictions)
        
        logger.info(f"Converted {len(raw_predictions)} predictions to legacy format")
        logger.info(f"Sample converted prediction: {compatible_predictions[0] if compatible_predictions else 'None'}")
        
        # 메트릭 정리
        metrics = prediction_state['latest_metrics']
        cleaned_metrics = {}
        if metrics:
            for key, value in metrics.items():
                cleaned_metrics[key] = safe_serialize_value(value)
        
        # 구간 점수 안전 정리
        interval_scores = prediction_state['latest_interval_scores'] or []
        cleaned_interval_scores = clean_interval_scores_safe(interval_scores)
        
        # MA 결과 정리 및 필요시 재계산
        ma_results = prediction_state['latest_ma_results'] or {}
        cleaned_ma_results = {}
        
        # 이동평균 결과가 없거나 비어있다면 재계산 시도
        if not ma_results or len(ma_results) == 0:
            logger.info("🔄 MA results missing, attempting to recalculate...")
            try:
                # 현재 데이터와 예측 결과를 사용하여 이동평균 재계산
                current_date = prediction_state.get('current_date')
                if current_date and prediction_state.get('latest_file_path'):
                    # 원본 데이터 로드
                    df = load_data(prediction_state['latest_file_path'])
                    if df is not None and not df.empty:
                        # 현재 날짜를 datetime으로 변환
                        if isinstance(current_date, str):
                            current_date_dt = pd.to_datetime(current_date)
                        else:
                            current_date_dt = current_date
                        
                        # 과거 데이터 추출
                        historical_data = df[df.index <= current_date_dt].copy()
                        
                        # 예측 데이터를 이동평균 계산용으로 변환
                        ma_input_data = []
                        for pred in raw_predictions:
                            try:
                                ma_item = {
                                    'Date': pd.to_datetime(pred.get('Date') or pred.get('date')),
                                    'Prediction': safe_serialize_value(pred.get('Prediction') or pred.get('prediction')),
                                    'Actual': safe_serialize_value(pred.get('Actual') or pred.get('actual'))
                                }
                                ma_input_data.append(ma_item)
                            except Exception as e:
                                logger.warning(f"⚠️ Error processing MA data item: {str(e)}")
                                continue
                        
                        # 이동평균 계산
                        if ma_input_data:
                            ma_results = calculate_moving_averages_with_history(
                                ma_input_data, historical_data, target_col='MOPJ'
                            )
                            if ma_results:
                                logger.info(f"✅ MA recalculated successfully with {len(ma_results)} windows")
                                prediction_state['latest_ma_results'] = ma_results
                            else:
                                logger.warning("⚠️ MA recalculation returned empty results")
                        else:
                            logger.warning("⚠️ No valid input data for MA calculation")
                    else:
                        logger.warning("⚠️ Unable to load original data for MA calculation")
                else:
                    logger.warning("⚠️ Missing current_date or file_path for MA calculation")
            except Exception as e:
                logger.error(f"❌ Error recalculating MA: {str(e)}")
        
        # MA 결과 정리
        for key, value in ma_results.items():
            if isinstance(value, list):
                cleaned_ma_results[key] = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_item = {}
                        for k, v in item.items():
                            cleaned_item[k] = safe_serialize_value(v)
                        cleaned_ma_results[key].append(cleaned_item)
                    else:
                        cleaned_ma_results[key].append(safe_serialize_value(item))
            else:
                cleaned_ma_results[key] = safe_serialize_value(value)
        
        # 어텐션 데이터 정리
        attention_data = prediction_state['latest_attention_data']
        cleaned_attention = None
        if attention_data:
            cleaned_attention = {}
            for key, value in attention_data.items():
                if key == 'image' and value:
                    cleaned_attention[key] = value  # base64 이미지는 그대로
                elif isinstance(value, dict):
                    cleaned_attention[key] = {}
                    for k, v in value.items():
                        cleaned_attention[key][k] = safe_serialize_value(v)
                else:
                    cleaned_attention[key] = safe_serialize_value(value)
        
        # 플롯 데이터 정리
        plots = prediction_state['latest_plots'] or {}
        cleaned_plots = {}
        for key, value in plots.items():
            if isinstance(value, dict):
                cleaned_plots[key] = {}
                for k, v in value.items():
                    if k == 'image' and v:
                        cleaned_plots[key][k] = v  # base64 이미지는 그대로
                    else:
                        cleaned_plots[key][k] = safe_serialize_value(v)
            else:
                cleaned_plots[key] = safe_serialize_value(value)
        
        response_data = {
            'success': True,
            'current_date': safe_serialize_value(prediction_state['current_date']),
            'predictions': compatible_predictions,  # 호환성 유지된 형태
            'interval_scores': cleaned_interval_scores,
            'ma_results': cleaned_ma_results,
            'attention_data': cleaned_attention,
            'plots': cleaned_plots,
            'metrics': cleaned_metrics if cleaned_metrics else None,
            'selected_features': prediction_state['selected_features'] or [],
            'feature_importance': safe_serialize_value(prediction_state.get('feature_importance')),
            'semimonthly_period': safe_serialize_value(prediction_state['semimonthly_period']),
            'next_semimonthly_period': safe_serialize_value(prediction_state['next_semimonthly_period'])
        }
        
        # JSON 직렬화 테스트
        try:
            test_json = json.dumps(response_data)
            logger.info(f"JSON serialization test: SUCCESS (length: {len(test_json)})")
        except Exception as json_error:
            logger.error(f"JSON serialization test: FAILED - {str(json_error)}")
            return jsonify({
                'success': False,
                'error': f'Data serialization error: {str(json_error)}'
            }), 500
        
        logger.info(f"=== Compatible Response Summary ===")
        logger.info(f"Total predictions: {len(compatible_predictions)}")
        logger.info(f"Has metrics: {cleaned_metrics is not None}")
        logger.info(f"Sample prediction fields: {list(compatible_predictions[0].keys()) if compatible_predictions else 'None'}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error creating compatible response: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Error creating response: {str(e)}'}), 500
    
@app.route('/api/clear-cache', methods=['POST'])
def clear_prediction_cache():
    """예측 캐시 클리어 (테스트용)"""
    global prediction_state
    
    try:
        # 상태 초기화
        prediction_state['latest_predictions'] = None
        prediction_state['latest_interval_scores'] = None
        prediction_state['latest_ma_results'] = None
        prediction_state['latest_attention_data'] = None
        prediction_state['latest_plots'] = None
        prediction_state['latest_metrics'] = None
        prediction_state['current_date'] = None
        prediction_state['selected_features'] = None
        prediction_state['feature_importance'] = None
        
        # 캐시 파일들도 삭제 (선택적)
        cache_dir = Path(PREDICTIONS_DIR)
        if cache_dir.exists():
            for file in cache_dir.glob("prediction_*.csv"):
                file.unlink()
            for file in cache_dir.glob("prediction_*.json"):
                file.unlink()
            logger.info("Cache files cleared")
        
        logger.info("Prediction cache cleared successfully")
        
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully'
        })
        
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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

@app.route('/api/results/plots', methods=['GET'])
def get_plots():
    """시각화 결과만 조회 API"""
    global prediction_state
    
    if prediction_state['latest_plots'] is None:
        return jsonify({'error': 'No plot data available'}), 404
    
    return jsonify({
        'success': True,
        'current_date': prediction_state['current_date'],
        'plots': prediction_state['latest_plots']
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
        logger.warning("No moving average results available")
        return jsonify({'error': 'No moving average results available'}), 404
    
    logger.info(f"Returning MA results with {len(prediction_state['latest_ma_results'])} windows")
    
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
            '/api/holidays',
            '/api/holidays/upload',
            '/api/holidays/reload',
            '/api/file/metadata',
            '/api/data/dates',
            '/api/predict',
            '/api/predict/accumulated',
            '/api/predict/status',
            '/api/results',
            '/api/results/predictions',
            '/api/results/interval-scores',
            '/api/results/moving-averages',
            '/api/results/attention-map',
            '/api/results/accumulated',
            '/api/results/accumulated/interval-scores',
            '/api/results/accumulated/<date>',
            '/api/results/accumulated/report',
            '/api/results/accumulated/visualization',
            '/api/results/reliability',  # 새로 추가된 신뢰도 API
            '/api/features'
        ],
        'new_features': [
            'Prediction consistency scoring (예측 신뢰도)',
            'Purchase reliability percentage (구매 신뢰도)',
            'Holiday management system',
            'Accumulated predictions analysis'
        ]
    })

# 4. API 엔드포인트 추가 - 누적 예측 시작
@app.route('/api/predict/accumulated', methods=['POST'])
def start_accumulated_prediction():
    """여러 날짜에 대한 누적 예측 시작 API (저장/로드 기능 포함)"""
    global prediction_state
    
    if prediction_state['is_predicting']:
        return jsonify({
            'success': False,
            'error': 'Prediction already in progress',
            'progress': prediction_state['prediction_progress']
        }), 409
    
    data = request.json
    filepath = data.get('filepath')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    save_to_csv = data.get('save_to_csv', True)
    use_saved_data = data.get('use_saved_data', True)  # 저장된 데이터 활용 여부
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Invalid file path'}), 400
    
    if not start_date:
        return jsonify({'error': 'Start date is required'}), 400
    
    # 백그라운드에서 누적 예측 실행
    thread = Thread(target=run_accumulated_predictions_with_save, 
                   args=(filepath, start_date, end_date, save_to_csv, use_saved_data))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Accumulated prediction started',
        'save_to_csv': save_to_csv,
        'use_saved_data': use_saved_data,
        'status_url': '/api/predict/status'
    })

# 5. API 엔드포인트 추가 - 누적 예측 결과 조회
@app.route('/api/results/accumulated', methods=['GET'])
def get_accumulated_results():
    global prediction_state
    
    logger.info("🔍 [ACCUMULATED] API call received")
    
    if prediction_state['is_predicting']:
        logger.warning("⚠️ [ACCUMULATED] Prediction still in progress")
        return jsonify({
            'success': False,
            'error': 'Prediction in progress',
            'progress': prediction_state['prediction_progress']
        }), 409

    if not prediction_state['accumulated_predictions']:
        logger.error("❌ [ACCUMULATED] No accumulated prediction results available")
        return jsonify({'error': 'No accumulated prediction results available'}), 404

    logger.info("✅ [ACCUMULATED] Processing accumulated predictions...")
    
    # 누적 구매 신뢰도 계산 - 올바른 방식 사용
    accumulated_purchase_reliability = calculate_accumulated_purchase_reliability(
        prediction_state['accumulated_predictions']
    )
    
    logger.info(f"💰 [ACCUMULATED] Purchase reliability calculated: {accumulated_purchase_reliability}")
    
    # 데이터 안전성 검사
    safe_interval_scores = []
    if prediction_state.get('accumulated_interval_scores'):
        safe_interval_scores = [
            item for item in prediction_state['accumulated_interval_scores'] 
            if item is not None and isinstance(item, dict) and 'days' in item and item['days'] is not None
        ]
        logger.info(f"📊 [ACCUMULATED] Safe interval scores count: {len(safe_interval_scores)}")
    else:
        logger.warning("⚠️ [ACCUMULATED] No accumulated_interval_scores found")
    
    consistency_scores = prediction_state.get('accumulated_consistency_scores', {})
    logger.info(f"🎯 [ACCUMULATED] Consistency scores keys: {list(consistency_scores.keys())}")
    
    # ✅ 캐시 통계 정보 추가
    cache_stats = prediction_state.get('cache_statistics', {
        'total_dates': 0,
        'cached_dates': 0,
        'new_predictions': 0,
        'cache_hit_rate': 0.0
    })
    
    response_data = {
        'success': True,
        'prediction_dates': prediction_state.get('prediction_dates', []),
        'accumulated_metrics': prediction_state.get('accumulated_metrics', {}),
        'predictions': prediction_state['accumulated_predictions'],
        'accumulated_interval_scores': safe_interval_scores,
        'accumulated_consistency_scores': consistency_scores,
        'accumulated_purchase_reliability': accumulated_purchase_reliability,
        'cache_statistics': cache_stats  # ✅ 캐시 통계 추가
    }
    
    logger.info(f"📤 [ACCUMULATED] Response summary: predictions={len(response_data['predictions'])}, metrics_keys={list(response_data['accumulated_metrics'].keys())}, reliability={response_data['accumulated_purchase_reliability']}")
    
    return jsonify(response_data)

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

def return_prediction_result(pred, date, match_type):
    """
    예측 결과를 API 응답 형식으로 반환하는 헬퍼 함수
    
    Parameters:
    -----------
    pred : dict
        예측 결과 딕셔너리
    date : str
        요청된 날짜
    match_type : str
        매칭 방식 설명
    
    Returns:
    --------
    JSON response
    """
    try:
        logger.info(f"🔄 [API] Returning prediction result for date={date}, match_type={match_type}")
        
        # 예측 데이터 안전하게 추출
        predictions = pred.get('predictions', [])
        if not isinstance(predictions, list):
            logger.warning(f"⚠️ [API] predictions is not a list: {type(predictions)}")
            predictions = []
        
        # 구간 점수 안전하게 추출 및 변환
        interval_scores = pred.get('interval_scores', {})
        if isinstance(interval_scores, dict):
            # 딕셔너리를 리스트로 변환
            interval_scores_list = []
            for key, interval in interval_scores.items():
                if interval and isinstance(interval, dict) and 'days' in interval:
                    interval_scores_list.append(interval)
            interval_scores = interval_scores_list
        elif not isinstance(interval_scores, list):
            logger.warning(f"⚠️ [API] interval_scores is neither dict nor list: {type(interval_scores)}")
            interval_scores = []
        
        # 메트릭 안전하게 추출
        metrics = pred.get('metrics', {})
        if not isinstance(metrics, dict):
            logger.warning(f"⚠️ [API] metrics is not a dict: {type(metrics)}")
            metrics = {}
        
        # 🔄 이동평균 데이터 추출 (캐시된 데이터 또는 파일에서 로드)
        ma_results = pred.get('ma_results', {})
        if not ma_results:
            # 파일별 캐시에서 MA 파일 로드 시도
            try:
                current_file = prediction_state.get('current_file')
                if current_file:
                    cache_dirs = get_file_cache_dirs(current_file)
                    predictions_dir = cache_dirs['predictions']
                    
                    pred_start_date = pred.get('prediction_start_date') or date
                    ma_file_path = predictions_dir / f"prediction_start_{pd.to_datetime(pred_start_date).strftime('%Y%m%d')}_ma.json"
                else:
                    # 백업: 글로벌 캐시 사용
                    pred_start_date = pred.get('prediction_start_date') or date
                    ma_file_path = Path(PREDICTIONS_DIR) / f"prediction_start_{pd.to_datetime(pred_start_date).strftime('%Y%m%d')}_ma.json"
                
                if ma_file_path.exists():
                    with open(ma_file_path, 'r', encoding='utf-8') as f:
                        ma_results = json.load(f)
                    logger.info(f"📊 [API] MA results loaded from file for {date}: {len(ma_results)} windows")
                else:
                    logger.info(f"⚠️ [API] No MA file found for {date}: {ma_file_path}")
                    
                    # 파일이 없으면 예측 데이터에서 재계산 (히스토리컬 데이터 없이 제한적으로)
                    if predictions:
                        ma_results = calculate_moving_averages_with_history(
                            predictions, None, target_col='MOPJ', windows=[5, 10, 23]
                        )
                        logger.info(f"📊 [API] MA results recalculated for {date}: {len(ma_results)} windows")
            except Exception as e:
                logger.warning(f"⚠️ [API] Error loading/calculating MA for {date}: {str(e)}")
                ma_results = {}
        
        # 🎯 Attention 데이터 추출
        attention_data = pred.get('attention_data', {})
        if not attention_data:
            # 파일별 캐시에서 Attention 파일 로드 시도
            try:
                current_file = prediction_state.get('current_file')
                if current_file:
                    cache_dirs = get_file_cache_dirs(current_file)
                    predictions_dir = cache_dirs['predictions']
                    
                    pred_start_date = pred.get('prediction_start_date') or date
                    attention_file_path = predictions_dir / f"prediction_start_{pd.to_datetime(pred_start_date).strftime('%Y%m%d')}_attention.json"
                else:
                    # 백업: 글로벌 캐시 사용
                    pred_start_date = pred.get('prediction_start_date') or date
                    attention_file_path = Path(PREDICTIONS_DIR) / f"prediction_start_{pd.to_datetime(pred_start_date).strftime('%Y%m%d')}_attention.json"
                
                if attention_file_path.exists():
                    with open(attention_file_path, 'r', encoding='utf-8') as f:
                        attention_data = json.load(f)
                    logger.info(f"📊 [API] Attention data loaded from file for {date}")
                else:
                    logger.info(f"⚠️ [API] No attention file found for {date}: {attention_file_path}")
            except Exception as e:
                logger.warning(f"⚠️ [API] Error loading attention data for {date}: {str(e)}")
        
        response_data = {
            'success': True,
            'date': date,
            'predictions': predictions,
            'interval_scores': interval_scores,
            'metrics': metrics,
            'ma_results': ma_results,  # 🔑 이동평균 데이터 추가
            'attention_data': attention_data,  # 🔑 Attention 데이터 추가
            'next_semimonthly_period': pred.get('next_semimonthly_period'),
            'actual_business_days': pred.get('actual_business_days'),
            'match_type': match_type,
            'data_end_date': pred.get('date'),  # 데이터 기준일 추가
            'prediction_start_date': pred.get('prediction_start_date')  # 예측 시작일 추가
        }
        
        logger.info(f"✅ [API] Successfully prepared response for {date}: predictions={len(predictions)}, interval_scores={len(interval_scores)}, ma_windows={len(ma_results)}, attention_data={bool(attention_data)}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"💥 [API] Error in return_prediction_result for {date}: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Error processing prediction result: {str(e)}',
            'date': date
        }), 500

# 8. API 엔드포인트 추가 - 특정 날짜 예측 결과 조회

@app.route('/api/results/accumulated/<date>', methods=['GET'])
def get_accumulated_result_by_date(date):
    """특정 날짜의 누적 예측 결과 조회 API"""
    global prediction_state
    
    logger.info(f"🔍 [API] Searching for accumulated result by date: {date}")
    
    if not prediction_state['accumulated_predictions']:
        logger.warning("❌ [API] No accumulated prediction results available")
        return jsonify({'error': 'No accumulated prediction results available'}), 404
    
    logger.info(f"📊 [API] Available prediction dates (data_end_date): {[p['date'] for p in prediction_state['accumulated_predictions']]}")
    
    # ✅ 1단계: 정확한 데이터 기준일 매칭 우선 확인
    logger.info(f"🔍 [API] Step 1: Looking for EXACT data_end_date match for {date}")
    for i, pred in enumerate(prediction_state['accumulated_predictions']):
        data_end_date = pred.get('date')  # 데이터 기준일
        
        logger.info(f"🔍 [API] Checking prediction {i+1}: data_end_date={data_end_date}")
        
        if data_end_date == date:
            logger.info(f"✅ [API] Found prediction by EXACT DATA END DATE match: {date}")
            logger.info(f"📊 [API] Prediction data preview: predictions={len(pred.get('predictions', []))}, interval_scores={len(pred.get('interval_scores', {}))}")
            return return_prediction_result(pred, date, "exact data end date")
    
    # ✅ 2단계: 정확한 매칭이 없으면 계산된 예측 시작일로 매칭
    logger.info(f"🔍 [API] Step 2: No exact match found. Looking for calculated prediction start date match for {date}")
    for i, pred in enumerate(prediction_state['accumulated_predictions']):
        data_end_date = pred.get('date')  # 데이터 기준일
        prediction_start_date = pred.get('prediction_start_date')  # 예측 시작일
        
        logger.info(f"🔍 [API] Checking prediction {i+1}: data_end_date={data_end_date}, prediction_start_date={prediction_start_date}")
        
        if data_end_date:
            try:
                data_end_dt = pd.to_datetime(data_end_date)
                calculated_start_date = data_end_dt + pd.Timedelta(days=1)
                
                # 주말과 휴일 건너뛰기
                while calculated_start_date.weekday() >= 5 or is_holiday(calculated_start_date):
                    calculated_start_date += pd.Timedelta(days=1)
                
                calculated_start_str = calculated_start_date.strftime('%Y-%m-%d')
                
                if calculated_start_str == date:
                    logger.info(f"✅ [API] Found prediction by CALCULATED PREDICTION START DATE: {date} (from data end date: {data_end_date})")
                    return return_prediction_result(pred, date, "calculated prediction start date from data end date")
                    
            except Exception as e:
                logger.warning(f"⚠️ [API] Error calculating prediction start date for {data_end_date}: {str(e)}")
                continue
    
    logger.error(f"❌ [API] No prediction results found for date {date}")
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

# 새로운 API 엔드포인트 추가
@app.route('/api/results/reliability', methods=['GET'])
def get_reliability_scores():
    """신뢰도 점수 조회 API"""
    global prediction_state
    
    # 단일 예측 신뢰도
    single_reliability = {}
    if prediction_state.get('latest_interval_scores') and prediction_state.get('latest_predictions'):
        try:
            # 실제 영업일 수 계산
            actual_business_days = len([p for p in prediction_state['latest_predictions'] 
                                       if p.get('Date') and not p.get('is_synthetic', False)])
            
            single_reliability = {
                'period': prediction_state['next_semimonthly_period']
            }
        except Exception as e:
            logger.error(f"Error calculating single prediction reliability: {str(e)}")
            single_reliability = {'error': 'Unable to calculate single prediction reliability'}
    
    # 누적 예측 신뢰도 (안전한 접근)
    accumulated_reliability = prediction_state.get('accumulated_consistency_scores', {})
    
    return jsonify({
        'success': True,
        'single_prediction_reliability': single_reliability,
        'accumulated_prediction_reliability': accumulated_reliability
    })

@app.route('/api/cache/clear/accumulated', methods=['POST'])
def clear_accumulated_cache():
    """누적 예측 캐시 클리어"""
    global prediction_state
    
    try:
        # 누적 예측 관련 상태 클리어
        prediction_state['accumulated_predictions'] = []
        prediction_state['accumulated_metrics'] = {}
        prediction_state['accumulated_interval_scores'] = []
        prediction_state['accumulated_consistency_scores'] = {}
        prediction_state['accumulated_purchase_reliability'] = 0
        prediction_state['prediction_dates'] = []
        
        logger.info("🧹 [CACHE] Accumulated prediction cache cleared")
        
        return jsonify({
            'success': True,
            'message': 'Accumulated prediction cache cleared successfully'
        })
        
    except Exception as e:
        logger.error(f"❌ [CACHE] Error clearing accumulated cache: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug/reliability', methods=['GET'])
def debug_reliability_calculation():
    """구매 신뢰도 계산 디버깅 API"""
    global prediction_state
    
    if not prediction_state['accumulated_predictions']:
        return jsonify({'error': 'No accumulated prediction results available'}), 404
    
    predictions = prediction_state['accumulated_predictions']
    print(f"🔍 [DEBUG] Total predictions: {len(predictions)}")
    
    debug_data = {
        'prediction_count': len(predictions),
        'predictions_details': []
    }
    
    total_score = 0
    
    for i, pred in enumerate(predictions):
        pred_date = pred.get('date')
        interval_scores = pred.get('interval_scores', {})
        
        print(f"📊 [DEBUG] Prediction {i+1} ({pred_date}):")
        print(f"   - interval_scores type: {type(interval_scores)}")
        print(f"   - interval_scores keys: {list(interval_scores.keys()) if isinstance(interval_scores, dict) else 'N/A'}")
        
        pred_detail = {
            'date': pred_date,
            'interval_scores_type': str(type(interval_scores)),
            'interval_scores_keys': list(interval_scores.keys()) if isinstance(interval_scores, dict) else [],
            'individual_scores': [],
            'best_score': 0
        }
        
        if isinstance(interval_scores, dict):
            for key, score_data in interval_scores.items():
                print(f"   - {key}: {score_data}")
                if isinstance(score_data, dict) and 'score' in score_data:
                    score_value = score_data.get('score', 0)
                    pred_detail['individual_scores'].append({
                        'key': key,
                        'score': score_value,
                        'full_data': score_data
                    })
                    print(f"     -> score: {score_value}")
        
        if pred_detail['individual_scores']:
            best_score = max([s['score'] for s in pred_detail['individual_scores']])
            pred_detail['best_score'] = best_score
            total_score += min(best_score, 3.0)  # 3점 제한
            print(f"   - Best score: {best_score}")
        
        debug_data['predictions_details'].append(pred_detail)
    
    max_possible_score = len(predictions) * 3
    reliability = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
    
    debug_data.update({
        'total_score': total_score,
        'max_possible_score': max_possible_score,
        'reliability_percentage': reliability
    })
    
    print(f"🎯 [DEBUG] CALCULATION SUMMARY:")
    print(f"   - Total predictions: {len(predictions)}")
    print(f"   - Total score: {total_score}")
    print(f"   - Max possible score: {max_possible_score}")
    print(f"   - Reliability: {reliability:.1f}%")
    
    return jsonify(debug_data)

@app.route('/api/cache/check', methods=['POST'])
def check_cached_predictions():
    """누적 예측 범위에서 캐시된 예측이 얼마나 있는지 확인"""
    data = request.get_json()
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    if not start_date or not end_date:
        return jsonify({'error': 'start_date and end_date are required'}), 400
    
    try:
        logger.info(f"🔍 [CACHE_CHECK] Checking cache availability for {start_date} to {end_date}")
        
        # 저장된 예측 확인
        cached_predictions = load_accumulated_predictions_from_csv(start_date, end_date)
        
        # 전체 범위 계산 (데이터 기준일 기준)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # 사용 가능한 날짜 계산 (데이터 기준일)
        available_dates = []
        current_dt = start_dt
        while current_dt <= end_dt:
            # 영업일만 포함 (주말과 휴일 제외)
            if current_dt.weekday() < 5 and not is_holiday(current_dt):
                available_dates.append(current_dt.strftime('%Y-%m-%d'))
            current_dt += pd.Timedelta(days=1)
        
        # 캐시된 날짜 목록
        cached_dates = [pred['date'] for pred in cached_predictions]
        missing_dates = [date for date in available_dates if date not in cached_dates]
        
        cache_percentage = round(len(cached_predictions) / max(len(available_dates), 1) * 100, 1)
        
        logger.info(f"📊 [CACHE_CHECK] Cache status: {len(cached_predictions)}/{len(available_dates)} ({cache_percentage}%)")
        
        return jsonify({
            'success': True,
            'total_dates_in_range': len(available_dates),
            'cached_predictions': len(cached_predictions),
            'cached_dates': cached_dates,
            'missing_dates': missing_dates,
            'cache_percentage': cache_percentage,
            'will_use_cache': len(cached_predictions) > 0,
            'estimated_time_savings': f"약 {len(cached_predictions) * 3}분 절약 예상" if len(cached_predictions) > 0 else "없음"
        })
        
    except Exception as e:
        logger.error(f"❌ [CACHE_CHECK] Error checking cached predictions: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/accumulated/recent', methods=['GET'])
def get_recent_accumulated_results():
    """
    페이지 로드 시 최근 누적 예측 결과를 자동으로 복원하는 API
    """
    try:
        # 저장된 예측 목록 조회 (최근 것부터)
        predictions_list = get_saved_predictions_list(limit=50)
        
        if not predictions_list:
            return jsonify({
                'success': False, 
                'message': 'No saved predictions found',
                'has_recent_results': False
            })
        
        # 날짜별로 그룹화하여 연속된 범위 찾기
        dates_by_groups = {}
        for pred in predictions_list:
            data_end_date = pred.get('data_end_date', pred.get('prediction_date'))
            if data_end_date:
                date_obj = pd.to_datetime(data_end_date)
                # 주차별로 그룹화 (같은 주의 예측들을 하나의 범위로 간주)
                week_key = date_obj.strftime('%Y-W%U')
                if week_key not in dates_by_groups:
                    dates_by_groups[week_key] = []
                dates_by_groups[week_key].append({
                    'date': data_end_date,
                    'date_obj': date_obj,
                    'pred_info': pred
                })
        
        # 가장 최근 그룹 선택
        if not dates_by_groups:
            return jsonify({
                'success': False, 
                'message': 'No valid date groups found',
                'has_recent_results': False
            })
        
        # 최근 주의 예측들 가져오기
        latest_week = max(dates_by_groups.keys())
        latest_group = dates_by_groups[latest_week]
        latest_group.sort(key=lambda x: x['date_obj'])
        
        # 연속된 날짜 범위 찾기
        start_date = latest_group[0]['date_obj']
        end_date = latest_group[-1]['date_obj']
        
        logger.info(f"🔄 [AUTO_RESTORE] Found recent accumulated predictions: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        
        # 기존 캐시에서 누적 결과 로드
        loaded_predictions = load_accumulated_predictions_from_csv(start_date, end_date)
        
        if not loaded_predictions:
            return jsonify({
                'success': False, 
                'message': 'Failed to load cached predictions',
                'has_recent_results': False
            })
        
        # 누적 메트릭 계산
        accumulated_metrics = {
            'f1': 0.0,
            'accuracy': 0.0,
            'mape': 0.0,
            'weighted_score': 0.0,
            'total_predictions': 0
        }
        
        for pred in loaded_predictions:
            metrics = pred.get('metrics', {})
            if isinstance(metrics, dict):
                accumulated_metrics['f1'] += metrics.get('f1', 0.0)
                accumulated_metrics['accuracy'] += metrics.get('accuracy', 0.0)
                accumulated_metrics['mape'] += metrics.get('mape', 0.0)
                accumulated_metrics['weighted_score'] += metrics.get('weighted_score', 0.0)
                accumulated_metrics['total_predictions'] += 1
        
        if accumulated_metrics['total_predictions'] > 0:
            count = accumulated_metrics['total_predictions']
            accumulated_metrics['f1'] /= count
            accumulated_metrics['accuracy'] /= count
            accumulated_metrics['mape'] /= count
            accumulated_metrics['weighted_score'] /= count
        
        # 구간 점수 계산
        accumulated_interval_scores = {}
        for pred in loaded_predictions:
            interval_scores = pred.get('interval_scores', {})
            if isinstance(interval_scores, dict):
                for interval in interval_scores.values():
                    if not interval or not isinstance(interval, dict) or 'days' not in interval or interval['days'] is None:
                        continue
                    interval_key = f"{format_date(interval['start_date'])}-{format_date(interval['end_date'])}"
                    if interval_key in accumulated_interval_scores:
                        accumulated_interval_scores[interval_key]['score'] += interval['score']
                        accumulated_interval_scores[interval_key]['count'] += 1
                    else:
                        accumulated_interval_scores[interval_key] = interval.copy()
                        accumulated_interval_scores[interval_key]['count'] = 1
        
        # 정렬된 구간 점수 리스트
        accumulated_scores_list = [
            v for v in accumulated_interval_scores.values()
            if v is not None and isinstance(v, dict) and 'days' in v and v['days'] is not None
        ]
        accumulated_scores_list.sort(key=lambda x: x['score'], reverse=True)
        
        # 구매 신뢰도 계산
        accumulated_purchase_reliability = calculate_accumulated_purchase_reliability(loaded_predictions)
        
        # 일관성 점수 계산
        unique_periods = set()
        for pred in loaded_predictions:
            if 'next_semimonthly_period' in pred and pred['next_semimonthly_period']:
                unique_periods.add(pred['next_semimonthly_period'])
        
        accumulated_consistency_scores = {}
        for period in unique_periods:
            try:
                consistency_data = calculate_prediction_consistency(loaded_predictions, period)
                accumulated_consistency_scores[period] = consistency_data
            except Exception as e:
                logger.error(f"Error calculating consistency for period {period}: {str(e)}")
        
        # 캐시 통계
        cache_statistics = {
            'total_dates': len(loaded_predictions),
            'cached_dates': len(loaded_predictions),
            'new_predictions': 0,
            'cache_hit_rate': 100.0
        }
        
        # 전역 상태 업데이트 (선택적)
        global prediction_state
        prediction_state['accumulated_predictions'] = loaded_predictions
        prediction_state['accumulated_metrics'] = accumulated_metrics
        prediction_state['prediction_dates'] = [p['date'] for p in loaded_predictions]
        prediction_state['accumulated_interval_scores'] = accumulated_scores_list
        prediction_state['accumulated_consistency_scores'] = accumulated_consistency_scores
        prediction_state['accumulated_purchase_reliability'] = accumulated_purchase_reliability
        prediction_state['cache_statistics'] = cache_statistics
        
        logger.info(f"✅ [AUTO_RESTORE] Successfully restored {len(loaded_predictions)} accumulated predictions")
        
        return jsonify({
            'success': True,
            'has_recent_results': True,
            'restored_range': {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'prediction_count': len(loaded_predictions)
            },
            'prediction_dates': [p['date'] for p in loaded_predictions],
            'accumulated_metrics': accumulated_metrics,
            'predictions': loaded_predictions,
            'accumulated_interval_scores': accumulated_scores_list,
            'accumulated_consistency_scores': accumulated_consistency_scores,
            'accumulated_purchase_reliability': accumulated_purchase_reliability,
            'cache_statistics': cache_statistics,
            'message': f"최근 누적 예측 결과를 자동으로 복원했습니다 ({start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})"
        })
        
    except Exception as e:
        logger.error(f"❌ [AUTO_RESTORE] Error restoring recent accumulated results: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False, 
            'error': str(e),
            'has_recent_results': False
        }), 500

@app.route('/api/cache/rebuild-index', methods=['POST'])
def rebuild_predictions_index_api():
    """예측 인덱스 재생성 API (rebuild_index.py 기능을 통합)"""
    try:
        # 현재 파일의 캐시 디렉토리 가져오기
        current_file = prediction_state.get('current_file')
        if not current_file:
            return jsonify({'success': False, 'error': '현재 업로드된 파일이 없습니다. 먼저 파일을 업로드해주세요.'})
        
        # 🔧 새로운 rebuild 함수 사용
        success = rebuild_predictions_index_from_existing_files()
        
        if success:
            cache_dirs = get_file_cache_dirs(current_file)
            index_file = cache_dirs['predictions'] / 'predictions_index.csv'
            
            # 결과 데이터 읽기
            index_data = []
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    index_data = list(reader)
            
            return jsonify({
                'success': True,
                'message': f'인덱스 파일을 성공적으로 재생성했습니다. ({len(index_data)}개 항목)',
                'file_location': str(index_file),
                'entries_count': len(index_data),
                'rebuilt_entries': [{'date': row.get('prediction_start_date', ''), 'data_end': row.get('data_end_date', '')} for row in index_data]
            })
        else:
            return jsonify({
                'success': False,
                'error': '인덱스 재생성에 실패했습니다. 로그를 확인해주세요.'
            })
        
    except Exception as e:
        logger.error(f"❌ Error rebuilding predictions index: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'인덱스 재생성 중 오류 발생: {str(e)}'})

# 메인 실행 부분 업데이트
if __name__ == '__main__':
    # 필요한 패키지 설치 확인
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna 패키지가 설치되어 있지 않습니다. 하이퍼파라미터 최적화를 위해 설치가 필요합니다.")
        logger.warning("pip install optuna 명령으로 설치할 수 있습니다.")
    
    # 🎯 파일별 캐시 시스템 - 레거시 디렉토리 및 인덱스 파일 생성 제거
    # 모든 데이터는 이제 파일별 캐시 디렉토리에 저장됩니다
    logger.info("🚀 Starting with file-based cache system - no legacy directories needed")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
