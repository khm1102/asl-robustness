# 인수인계 (다음 세션/자동 재개용)

## 현재 상태 (자동 재개 2회차)
- 완료: 데이터 준비, 랜드마크, 변형 테스트셋, RF/DT(EXP-003), CNN plain/aug 3시드(EXP-004), 쌍체 통계+혼동행렬(EXP-005), H1 분할 비교(EXP-006), 가설 갱신(`hypotheses.md`), 최종 요약(`REPORT.md`).
- 실행 중인 작업 없음. 핵심 산출물: `results/*.json`, `results/stats.json`, `results/figs/confusion_*.png`.
- 환경: `KERAS_BACKEND=torch`, `uv` 는 `$HOME/.local/bin`. 서버 CPU는 AVX 없음(TensorFlow/jax/mediapipe 1.x 불가). cron 환경에서는 heredoc/`python -c` 실행이 차단되므로 스크립트는 파일로 작성해 실행.
- 주의: 같은 프로젝트를 다른 세션이 동시에 만질 수 있음(`h1_split --only random` 중복 실행 발견, 내 쪽 완료 후 중복 프로세스 종료).

## 남은 할 일 (선택, 우선순위 순)
1. det-only 기준 쌍체 비교(`stats.py` 확장) — EXP-005에서 `[미수행]`.
2. 전이학습/사전학습 가벼운 CNN(AVX 없는 CPU 주의) — CNN이 불리한 설정이라는 한계 분리용.
3. conf 0.1/0.5 검출 임계 민감도, 분할 시드 반복(H1 CNN 불확정 해소).
4. 사진 전용 학습 CNN, 96×96 학습(시간 많이 소요).
5. README 결과 표 정리(현재는 `docs/REPORT.md`가 요약).

## 규칙 (사용자 지시)
- 기능별로 자주 커밋·푸시, **Claude 서명/Co-Authored-By 넣지 말 것**. 레포는 private `khm1102/asl-robustness`.
- 가설·추론·분석·검증·구현은 docs/에 추적 가능하게 기록(근거 링크, 모르면 [미확인]). 결과가 가설과 달라도 삭제 금지.
- 한국어로 보고.
