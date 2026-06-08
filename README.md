# Seoul Crowd Forecast API

서울 열린데이터광장의 서울 실시간 인구 API를 수집해 SQLite에 저장하고, 저장된 데이터를 PyTorch 모델 학습에 사용하는 FastAPI 프로젝트입니다.

## 121개 장소 수집

서울 실시간 도시데이터 공식 안내에 따르면 주요 121개 장소를 제공하고, API는 한 번에 1개 장소만 호출할 수 있으며, 장소명 또는 장소코드 중 하나로 조회할 수 있습니다.

이 프로젝트는 전체 수집 시 장소코드 `POI001`부터 `POI121`까지를 순차 호출합니다. API 응답에 실제 `AREA_NM`, `AREA_CD`가 포함되므로 DB에는 실제 장소명과 코드가 저장됩니다.

샘플키는 `광화문·덕수궁`만 조회할 수 있습니다. 121개 전체 수집은 서울 열린데이터광장 인증키가 필요합니다.

## 실행

```powershell
uvicorn app.main:app --reload
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## 주요 API

상태 확인:

```text
GET /health
```

121개 장소코드 확인:

```text
GET /areas
```

단일 장소 실시간 조회:

```text
GET /population/current?area=광화문·덕수궁
GET /population/current?area=POI009
```

단일 또는 복수 장소 저장:

```text
POST /collect?area=POI009
POST /collect?area=POI009&area=POI014
```

서울 주요 121개 장소를 한 번 저장:

```text
POST /collect/all
```

서울 주요 121개 장소를 5분 동안 반복 수집:

```text
POST /collect/all/continuous
```

수집 작업 상태 확인:

```text
GET /collect/all/continuous/status
```

저장된 데이터 조회:

```text
GET /observations?area=광화문·덕수궁&limit=100
```

학습:

```text
POST /train?epochs=100&learning_rate=0.01&train_ratio=0.7&validation_ratio=0.15
```

예측:

```text
GET /predictions?area=광화문·덕수궁&target_date=2026-06-15&target_time=18:30
```

## 참고

서울 실시간 API는 과거 데이터를 제공하지 않습니다. 모델 학습에 필요한 과거 데이터는 `/collect/all/continuous`를 주기적으로 호출해 직접 누적해야 합니다.

