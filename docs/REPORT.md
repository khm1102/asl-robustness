# 최종 요약: ASL 알파벳 인식 — CNN vs MediaPipe 손 관절 + Random Forest, 환경 변화 강건성

> 모든 수치는 `results/*.json`의 실제 실행 결과. 근거 문서는 `docs/experiments/EXP-001…007`, 결정은 `docs/decisions/0001…0005`, 가설 판정은 `docs/hypotheses.md`.

## 한 줄 결론
이 데이터·설정에서는 **손 관절(거리·각도 특징) + RF가 모든 합성 환경 변화에서 CNN보다 높은 정확도**를 보였고, 어두운 조명이 약해지는 구간에서는 RF의 손실이 분류가 아니라 **MediaPipe 검출 실패**에서 나온다. 단 CNN이 깨끗한 이미지에서도 낮아 "CNN이 환경에 약하다"로 일반화할 수는 없다.

## 설정
- 데이터: HF `Marxulia/asl_sign_languages_alphabets_v03` 중 10개 알파벳(A B D F I L U V W Y) 4,198장. (Kaggle 원본은 인증 부재로 사용 못 함; `decisions/0002`) 사진 3,554 / 손 삽화 644.
- 분할: dHash 군집 기준 그룹 분할 train 2,518 / val 840 / test 840. 주 평가는 test의 **사진 715장**(삽화 제외; `decisions/0005`).
- 모델: CNN-plain / CNN-aug (Keras 3 torch 백엔드, 64×64, 80,570 params, 3 시드); MediaPipe 0.10.14 landmark → 특징 3종(raw/norm/geom) → RF(300)·DT.
- 환경 변화(합성): 어두움 3단계, 가장자리 잡동사니, 이동, 크기 축소/확대, 회전 15°. 서버 CPU에 AVX가 없어 TensorFlow/jax/mediapipe 1.x 불가(`decisions/0001, 0004`).

## 핵심 결과 (사진 정확도, e2e = 검출 실패를 오답 처리)
| 조건 | RF geom | CNN-plain | CNN-aug |
|---|---|---|---|
| clean | 0.942 | 0.577 | 0.477 |
| dark_0.5 | 0.699 | 0.110 | 0.318 |
| dark_0.25 | 0.297 | 0.099 | 0.147 |
| dark_0.12 | 0.048 | 0.103 | 0.096 |
| clutter | 0.826 | 0.199 | 0.374 |
| shift | 0.859 | 0.429 | 0.423 |
| scale_small | 0.938 | 0.247 | 0.284 |
| scale_large | 0.699 | 0.439 | 0.386 |
| rot15 | 0.936 | 0.559 | 0.460 |

## 가설 판정 요약
- H1 분할 누수: **기각(RF)/판정불가(CNN)** — 이 데이터는 근접 중복이 약 1.3%로 적다 (EXP-001, 006).
- H2 어두운 조명에서 CNN이 RF보다 더 하락: **부분 지지** — dark_0.5에서만 명확, 더 어두우면 RF가 검출로 붕괴하고 CNN은 바닥에 닿아 역전 (EXP-005).
- H3 aug는 어둠은 좁히나 배경/잡음은 못 좁힘: **부분 기각** — aug가 clutter 드롭 격차를 지움. 단 clean이 9.9%p 하락하고, 증강에 같은 종류 변형이 들어 있어 공정 비교가 아님.
- H4 정규화 RF는 이동/크기에 거의 영향 없음: **부분 기각** — raw 대비 큰 개선(이동 하락 22.2→4.8%p)이나 "거의 0"은 아님 (EXP-003).
- H5 검출 실패가 RF e2e를 좌우: **지지** — 임계값 0.1에서도 유지 (EXP-003, 005, 007).
- H6 RF>DT, geom 최상: **지지** — geom이 norm보다 약 +10%p; RF−DT는 geom에서 2.3%p뿐 (EXP-003, 005).
- H7 삽화는 검출 못 함: **지지** — 2.9% vs 94.5% (EXP-002).

## 한계 (결론의 효력 범위)
1. 환경 변화는 합성이다. 실제 촬영 조명/배경이 아니다. 팀원 직접 촬영 테스트셋이 이 파이프라인의 `testsets` 슬롯에 들어가야 외적 타당성이 생긴다.
2. CNN이 약한 이유를 분리 못 함: 데이터량(클래스당 약 250장 학습), 64×64로 축소(연산 제약), 소형 모델, 하이퍼파라미터 미탐색. 전이학습/해상도 ablation은 하지 못함.
3. RF 우위의 상당 부분은 **특징 설계(거리·각도)**와 **외부 사전학습 모델(MediaPipe)** 덕이다. "머신러닝 vs 딥러닝" 비교가 아니라 "사전학습 손 검출기를 쓰는 파이프라인 vs 처음부터 학습한 소형 CNN" 비교다.
4. 통계: 시드 3개, 부트스트랩 CI가 시드 분산 미반영, 다중비교 보정 없음, 같은 촬영자 샘플 상관 무시.
5. 데이터 출처·라이선스·촬영자 ID 불명. 삽화 태그는 색 통계 휴리스틱 `[정확도 미검증]`.
6. 검출 전 밝기 보정 같은 RF 파이프라인 개선은 시험하지 않았다. 어두운 조건의 RF 붕괴는 이 보정으로 줄어들 수 있다 `[미확인]`.

## 재현
```bash
uv sync
uv run python -m aslrob.prepare && uv run python -m aslrob.landmarks && uv run python -m aslrob.testsets
uv run python -m aslrob.rf_exp && KERAS_BACKEND=torch uv run python -m aslrob.cnn_exp --variant plain && KERAS_BACKEND=torch uv run python -m aslrob.cnn_exp --variant aug
uv run python -m aslrob.stats
```
(데이터 parquet는 `data/hf_v03.parquet`로 직접 받아야 함; 레포에 포함하지 않음.)

## 발표(3~5분) 권장 메시지
1. 같은 손동작을 이미지로 읽을 때와 관절 좌표로 읽을 때, 환경이 바뀌면 어떻게 달라지는가.
2. 좌표 방식은 이동·크기·배경에 강하지만, **손 검출이 실패하는 어두운 환경에서는 같이 무너진다.**
3. CNN은 증강으로 어둠·배경은 보완되지만 비용(깨끗한 이미지 정확도 하락)이 있다.
4. 한계: 합성 변형, 소규모 데이터 — 직접 촬영 검증이 다음 단계.
