# 웹 데모 사용법

## 실행 (서버, Docker)
```bash
# 1) 모델 저장 (한 번)
KERAS_BACKEND=torch uv run python -m aslrob.train_final      # models/*.keras, rf_geom.joblib
# 2) 컨테이너
docker compose up -d --build
# 3) 접속 (자체 서명 https → 브라우저 경고에서 '고급 → 계속')
https://<서버IP>:8080
```
- 필요한 것: `data/processed/`(prepare·landmarks·testsets 산출물), `results/`, `models/`.
- 포트 8080만 사용, 메모리 1.3GB 제한.

## 로컬 실행 (본인 PC에서, 웹캠이 더 빠름)
```bash
uv sync
KERAS_BACKEND=torch uv run uvicorn aslrob.web.app:app --port 8000   # http://localhost:8000 은 카메라 허용
```

## 탭
1. **실시간 웹캠**: 초록 정사각형 안에 손. 환경 변화(합성) 드롭다운으로 어둡게/잡동사니/이동/크기/회전을 켜면 서버가 적용한 뒤 인식. 오른쪽에 모델이 본 입력과 손 관절, 아래 세 모델 예측(확률 막대)과 지연시간.
2. **테스트셋 둘러보기**: 실험에 쓴 변형 test 이미지를 조건/글자/“RF만 맞음”, “CNN만 맞음” 등으로 필터링, 정답과 세 모델 예측을 색으로 표시, 관절 오버레이.
3. **실험 결과**: 조건별 막대그래프(정확도 / clean 대비 하락 / 검출 성공만 / 검출률), 표, 분할 누수·검출 임계값 부가 실험, 혼동행렬.
4. **내 촬영 평가**: 라이브 탭에서 저장한 사진을 환경 태그별로 채점.
5. **문서**: REPORT, 가설 대장, EXP/결정 기록을 렌더링.

## 주의
- 라이브 예측은 학습 데이터(정적 사진 10개 알파벳: A B D F I L U V W Y) 범위만 맞다.
- 라이브 결과는 눈으로 보는 참고용이며 통계적 결과가 아니다. 표본이 쌓인 평가는 “내 촬영 평가” 탭 또는 `EXP-005` 참고.
