# 인수인계 (다음 세션/자동 재개용)

## 현재 상태
- 완료: 데이터 준비, 랜드마크, 변형 테스트셋, RF/DT 평가(EXP-003), CNN 사전 점검.
- **갱신(자동 재개 1회차)**: plain 완료, `docs/experiments/EXP-004` 초안 커밋. aug는 진행 중(로그 epoch 12/40 시점) → 완료되면 EXP-004에 aug 추가 후 2~6번 진행.
- 실행 중: `python -m aslrob.cnn_exp --variant plain|aug --seeds 3 --epochs 40` (순차). 로그 `/opt/data/cache/scratch/cnn_plain.out`, `cnn_aug.out`. 완료 시 `results/cnn_{plain,aug}_group.json`, `results/cnn_preds_*_group.npz` 생성.
- 환경: `KERAS_BACKEND=torch`, `uv` 는 `$HOME/.local/bin`. 서버 CPU는 AVX 없음(TensorFlow/jax/mediapipe 1.x 불가).

## 다음 할 일 (순서)
1. CNN 결과 확인 → `docs/experiments/EXP-004-cnn-robustness.md` (조건별 photo 정확도, 시드 sd, clean 대비 하락폭). 커밋·푸시.
2. `aslrob/stats.py`: RF(geom, norm) vs CNN-plain/aug 비교 — 같은 테스트 샘플 기준 paired bootstrap 95% CI, McNemar. e2e와 det-only 모두. 조건별 Accuracy Drop = clean − cond.
3. 혼동행렬(clean, dark_0.5, shift) 그림 `results/figs/`. 
4. H1 측정: `split_random` 으로 CNN/RF 재학습해 group 분할과 비교(누수 정도). RF는 `--split_kind random` (testsets가 group 테스트 기준이라 random용 테스트셋 필요 → 단순히 split 인덱스만 바꿔 clean 정확도 비교로 충분).
5. 추가 변형(시간 있으면): CNN 전이학습(MobileNetV3 대신 가벼운 것; AVX 없음 주의), conf 0.1/0.5 민감도, 사진만 학습한 CNN.
6. `docs/hypotheses.md` H1~H7 상태 갱신, `docs/REPORT.md` 최종 요약(한계 포함), README 결과 표.

## 규칙 (사용자 지시)
- 기능별로 자주 커밋·푸시, **Claude 서명/Co-Authored-By 넣지 말 것**. 레포는 private `khm1102/asl-robustness`.
- 가설·추론·분석·검증·구현은 docs/에 추적 가능하게 기록(근거 링크, 모르면 [미확인]). 결과가 가설과 달라도 삭제 금지.
- 서브에이전트와 토론하며 진행(결과 자동전달이 안 되면 /opt/data/cache/delegation/live/*/task-N.log 에서 읽기).
- 한국어로 보고.
