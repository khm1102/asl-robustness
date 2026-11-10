# 0002. 데이터셋: Kaggle 대신 HuggingFace 대체본으로 시작

- 상태: 잠정 채택(Kaggle 원본 확보 시 재검토)
- 맥락: 기획안은 Kaggle `grassknoted/asl-alphabet`(약 87k장, 200×200). 이 환경에는 Kaggle 인증 정보가 없음(`~/.kaggle` 없음).
- 관찰: Kaggle 원본은 단일 촬영자·단일 배경이라는 HF 모델 카드 설명이 있음(2차 정보, 직접 검증 안 함).
- 대체본: `Marxulia/asl_sign_languages_alphabets_v03`, 10,873장, 26클래스(클래스당 약 380~420), 평균 약 447×407. 샘플 격자 육안 확인: 배경·피부톤·조명이 다양, 일부 손 그림/글자 오버레이/프레임에서 손 잘림 포함.
- 한계: 출처·촬영자 ID 불명 → 근접 중복으로 인한 train/test 누수 가능성. [미확인] 라이선스.
- 후속: 중복 제거(perceptual hash) 후 분할(`H`-누수 점검).
