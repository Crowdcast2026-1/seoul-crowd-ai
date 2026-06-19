# Seoul Crowd Forecast API

서울 실시간 도시데이터 API의 인구 현황 데이터를 수집해서 SQLite에 저장하고, 저장된 과거 데이터를 기반으로 사용자가 선택한 미래 날짜와 시간대의 혼잡도를 예측하는 FastAPI 프로젝트입니다.

예측 대상은 인구 수가 아니라 서울 API가 제공하는 혼잡도 라벨입니다.

```text
여유, 보통, 약간 붐빔, 붐빔
```

## 데이터 수집

서울 API는 주요 장소를 `POI001`부터 `POI121`까지의 장소 코드로 조회할 수 있습니다. 이 프로젝트는 전체 수집 시 121개 코드를 순회하면서 API를 호출하고, 응답에 포함된 실제 장소명과 장소 코드를 SQLite에 저장합니다.

```text
POST /collect/all
```

5분마다 계속 수집하려면 아래 API를 호출합니다. 한 번 시작하면 stop API를 호출하거나 서버 프로세스를 종료할 때까지 계속 동작합니다.

```text
POST /collect/all/continuous
```

수집 주기를 바꾸려면 `interval_minutes`를 지정합니다.

```text
POST /collect/all/continuous?interval_minutes=10
```

상태 확인과 중지는 아래 API를 사용합니다.

```text
GET  /collect/all/continuous/status
POST /collect/all/continuous/stop
```

## 학습과 예측

학습은 저장된 SQLite 데이터를 읽어서 PyTorch 모델을 훈련합니다.

```text
POST /train?epochs=500&learning_rate=0.1
```

예측은 장소명, 미래 날짜, 미래 시간을 입력받습니다.

```text
GET /predictions?area=광화문·덕수궁&target_date=2026-06-20&target_time=18:30
```

학습/검증/테스트 분할 비율은 API 입력으로 받지 않고 아래 값으로 고정합니다.

```text
Train      70%
Validation 15%
Test       15%
```

과적합을 줄이기 위해 학습은 마지막 epoch 모델을 그대로 저장하지 않습니다. 매 epoch validation loss를 확인하고, validation loss가 가장 낮았던 epoch의 모델을 최종 모델로 저장합니다.

기본 학습 안정화 설정:

- optimizer: AdamW
- learning rate: 0.1
- weight decay: 0.0001
- early stopping patience: 40 epoch
- LR scheduler: validation loss가 15 epoch 동안 개선되지 않으면 learning rate를 0.5배로 감소
- minimum learning rate: 0.0001

따라서 `epochs=500`으로 실행해도 후반부에 validation 성능이 나빠지면 best validation checkpoint가 저장되고, 개선이 오래 멈추면 early stopping으로 학습을 종료합니다.

## 혼잡도 분류를 직접 예측하는 이유

이 프로젝트는 "인구 수를 예측한 뒤 임계값으로 혼잡도를 나누는 방식"이 아니라, 처음부터 혼잡도 라벨을 분류합니다.

그 이유는 다음과 같습니다.

1. 서울 API가 이미 혼잡도 라벨을 제공합니다.

   API 응답에는 `여유`, `보통`, `약간 붐빔`, `붐빔` 같은 혼잡도 값이 포함됩니다. 즉 모델이 학습할 정답 라벨이 이미 존재합니다.

2. 최종 서비스 출력이 혼잡도입니다.

   사용자가 필요한 정보가 "몇 명인지"보다 "얼마나 붐비는지"라면, 인구 수를 중간에 예측하고 다시 라벨로 변환하는 과정은 불필요합니다. 중간 단계를 줄이면 오차가 누적될 가능성도 줄어듭니다.

3. 같은 인구 수라도 장소마다 혼잡도 기준이 다를 수 있습니다.

   예를 들어 같은 5,000명이라도 강남역과 작은 공원에서의 혼잡도는 다르게 해석될 수 있습니다. 혼잡도는 단순 인구 수뿐 아니라 장소 면적, 수용량, 평소 대비 증가량, 유동 패턴 등이 반영된 결과일 가능성이 큽니다.

4. API의 인구 수는 정확한 단일값이 아니라 범위값입니다.

   서울 API는 `AREA_PPLTN_MIN`, `AREA_PPLTN_MAX`처럼 최소/최대 범위를 제공합니다. 회귀 모델을 만들려면 중간값을 임의 정답으로 써야 하는데, 이 값은 실제 인구 수가 아니라 근사치입니다.

5. 서울시 내부 혼잡도 산정 기준을 정확히 알 수 없습니다.

   공식 임계값이 명확히 공개되어 있고 모든 장소에 동일하게 적용된다면 인구 수 예측 후 라벨 변환도 가능합니다. 하지만 현재는 API가 제공하는 라벨을 그대로 학습 대상으로 삼는 편이 더 직접적이고 안정적입니다.

주의할 점도 있습니다. 현재 데이터는 `여유` 라벨이 많기 때문에 accuracy만 보면 모델이 좋아 보일 수 있습니다. 학습 결과를 평가할 때는 class별 precision, recall, macro F1, confusion matrix를 같이 확인해야 합니다.

## 현재 모델 구조

모델은 `app/ml.py`의 `CrowdNet`입니다. 저장된 지역 수가 111곳이라면 입력 차원은 `111 + 7 = 118`입니다.

입력 피처:

- 지역 one-hot
- 시간 sin/cos
- 요일 sin/cos
- 월 sin/cos
- 주말 여부

네트워크 구조:

```text
Linear(input_dim -> 32)
ReLU
Dropout(0.1)
Linear(32 -> 32)
ReLU
Linear(32 -> 4)
```

출력 클래스:

```text
여유, 보통, 약간 붐빔, 붐빔
```

손실 함수는 cross entropy이고 optimizer는 AdamW입니다. Validation loss가 가장 낮았던 epoch의 모델을 저장하며, validation loss가 오래 개선되지 않으면 learning rate를 낮추고 early stopping을 적용합니다.

## 주요 API

```text
GET  /health
GET  /areas
GET  /areas/names
GET  /population/current?area=POI009
POST /collect?area=POI009
POST /collect/all
POST /collect/all/continuous
GET  /collect/all/continuous/status
POST /collect/all/continuous/stop
GET  /observations?area=광화문·덕수궁&limit=100
POST /train
GET  /predictions
```

## 실행

```powershell
uvicorn app.main:app --reload
```

Swagger 문서는 아래 주소에서 확인할 수 있습니다.

```text
http://127.0.0.1:8000/docs
```

## app 디렉터리 구조

### `app/__init__.py`

`app` 디렉터리를 Python 패키지로 인식시키는 초기화 파일입니다.

### `app/areas.py`

서울 주요 장소 코드 목록을 제공합니다.

- `SEOUL_MAJOR_AREA_CODES`: `POI001`부터 `POI121`까지의 코드
- `all_area_codes()`: 전체 장소 코드 리스트 반환
- `all_area_names()`: 코드 없이 장소명 리스트만 반환

### `app/collection_job.py`

백그라운드 연속 수집 작업을 관리합니다.

- 5분마다 전체 장소 수집
- 중복 실행 방지
- stop 요청 처리
- 라운드별 수집 성공/실패 수 기록
- 최근 오류와 이벤트 로그 저장

### `app/config.py`

환경 설정을 로드합니다.

- `.env` 파일 로드
- 서울 API key/base URL
- SQLite DB 경로
- 모델 저장 경로
- 기본 장소 설정
- API timeout 설정

### `app/database.py`

SQLite 저장소를 관리합니다.

- `population_observations` 테이블 생성
- API 응답 저장
- 저장된 관측값 조회
- 전체/장소별 행 수 조회
- `PopulationObservation` 객체와 DB row 변환

중복 저장 방지를 위해 `(area_name, observed_at)`에 unique 제약이 있습니다.

### `app/main.py`

FastAPI 엔트리포인트입니다.

- 서버 시작 시 DB 초기화
- API 라우트 정의
- 수집, 조회, 학습, 예측 요청 처리
- 내부 예외를 HTTP 응답으로 변환

### `app/ml.py`

PyTorch 모델 학습과 예측 로직입니다.

- 혼잡도 라벨 정의
- `CrowdNet` 모델 정의
- 학습/검증/테스트 분할
- feature 생성
- loss/accuracy 계산
- 모델 아티팩트 저장/로드
- 미래 날짜/시간 혼잡도 예측

### `app/models.py`

프로젝트에서 공유하는 데이터 모델입니다.

- `PopulationObservation`: 서울 API에서 받은 인구/혼잡도 관측값
- `TrainingConfig`: epochs, learning rate, 고정 split ratio, seed 설정

### `app/seoul_api.py`

서울 실시간 도시데이터 API 클라이언트입니다.

- API URL 생성
- XML 응답 요청
- XML 파싱
- API 오류 처리
- 인구 수 범위, 혼잡도, 성별/거주자 비율, 관측 시간 변환

### `app/workflows.py`

API 라우트와 세부 모듈 사이의 작업 흐름을 묶는 계층입니다.

- 서울 API 클라이언트 생성
- 단일/복수 장소 수집
- 121개 전체 장소 수집
- DB 데이터로 모델 학습
- 저장된 모델로 예측
