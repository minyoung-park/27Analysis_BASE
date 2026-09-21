# Segment Anything

Alexander Kirillov, Eric Mintun, Nikhila Ravi 외 (Meta AI, FAIR, 2023). arXiv:2304.02643.

보통 SAM이라고 부르는 논문. 세그멘테이션용 foundation model을 만들려고 task, model, dataset을 같이 낸다.

---

## 어떤 문제를 해결하려 했는가? (Abstract)

이미지 세그멘테이션을 위한 새 과제, 모델, 데이터셋을 제안한다. 모델을 데이터 수집 루프에 넣어서 지금까지 제일 큰 세그멘테이션 데이터셋 SA-1B를 만들었다. 마스크 10억 개 이상, 라이선스 있는 이미지 1100만 장.

모델은 promptable하게 학습해서 새 이미지 분포와 과제에 zero-shot으로 옮긴다. 여러 과제에서 완전 지도학습 결과와 비슷하거나 더 나은 경우도 있다. SAM이랑 SA-1B는 https://segment-anything.com 에서 공개한다.

---

## 연구 동기와 문제점은 무엇인가? (Introduction)

NLP는 웹 규모 데이터로 학습한 foundation model이 프롬프트만으로 zero-shot, few-shot을 한다. 비전에서는 CLIP, ALIGN처럼 이미지-텍스트 정렬이 그 역할을 한다. 근데 비전 문제는 그 범위 밖이 많고, 세그멘테이션처럼 라벨이 웹에 자연스럽게 안 쌓이는 과제가 있다.

목표는 세그멘테이션 foundation model이다. promptable 모델을 넓은 데이터로 사전학습하고, 새 분포에서는 프롬프트로 푼다. 성공하려면 세 가지가 같이 필요하다.

1. zero-shot을 가능하게 하는 task는 뭔가
2. 그에 맞는 모델 구조는 뭔가
3. 그 task와 모델을 먹일 데이터는 뭔가

세그멘테이션 마스크는 웹에 없다. 그래서 data engine을 만든다. 모델이 주석을 돕고, 새 데이터로 모델을 다시 학습하는 루프다.

---

## 관련 연구 동향은 어떠한가? (Related Works)

세그멘테이션은 종류가 많다. interactive, edge, superpixel, object proposal, semantic, instance, panoptic. 기존 multi-task 모델은 학습 때 본 과제 집합을 테스트에서도 그대로 한다. SAM은 프롬프트로 새 과제에 붙이는 쪽이다. 예를 들어 박스 detector 출력을 프롬프트로 주면 instance segmentation이 된다.

interactive segmentation(RITM, SimpleClick, FocalClick)은 사람이 점을 여러 번 찍어서 마스크를 맞춘다. SAM도 그 세팅을 시뮬레이션하지만, 목표는 입력이 애매해도 **항상** 그럴듯한 마스크를 내는 것이다.

CLIP은 텍스트 프롬프트로 분류기를 만든다. SAM은 점, 박스, 마스크, (실험적으로) 텍스트로 마스크를 만든다. 이미지 인코더는 MAE 사전학습 ViT를 쓴다. 텍스트 프롬프트는 CLIP 텍스트 인코더를 그대로 쓴다.

데이터 쪽은 COCO, LVIS, ADE20K, Open Images가 기존 큰 세트다. Open Images가 마스크 270만 개인데 SA-1B는 11억이라 약 400배다.

---

## 연구 접근법과 모델 구조는 어떻게 되는가? (Method)

세 덩어리다. task, SAM, data engine.

**Promptable segmentation**

프롬프트는 이 이미지에서 뭘 자를지다. foreground/background 점, 박스, 러프 마스크, 자유 텍스트. 모델은 유효한 마스크를 내야 한다. 점이 애매하면(셔츠 vs 사람) 그중 하나라도 말이 되면 된다. NLP에서 애매한 프롬프트에도 문장을 뱉는 것과 비슷하다.

사전학습은 마스크마다 점/박스/마스크 프롬프트를 시뮬레이션하고 GT와 비교한다. 11 round. interactive segmentation을 흉내 내되, 입력이 적어도 바로 유효한 마스크를 내는 게 목표다.

Zero-shot은 과제에 맞는 프롬프트를 만들면 된다. 고양이 박스 detector가 있으면 그 박스를 SAM에 넣어서 instance mask를 얻는다.

**SAM 구조**

무거운 이미지 인코더 한 번, 가벼운 prompt encoder + mask decoder를 프롬프트마다 돌린다. 이미지 임베딩을 재사용하니까 브라우저에서 프롬프트당 약 50ms.

- 이미지 인코더: MAE 사전학습 ViT. 해상도 높게 받도록 조금 바꿈. 이미지당 한 번.
- prompt encoder: 점/박스는 positional encoding + 타입 임베딩. 텍스트는 CLIP 텍스트 인코더. dense 마스크는 conv 후 이미지 임베딩에 더함.
- mask decoder: modified Transformer decoder 2블록. prompt self-attention, prompt↔image cross-attention. 올린 이미지 임베딩에 대해 output token을 MLP로 동적 선형분류기로 바꿔서 픽셀 foreground 확률을 낸다.

애매함: 출력을 하나면 여러 물체를 평균하게 된다. 그래서 마스크를 3개 낸다 (whole / part / subpart가 흔해서). 학습 때는 세 개 중 GT와 제일 가까운 것만 backprop한다 (min loss). 각 마스크에 IoU 예측 점수를 붙여서 순위를 매긴다.

손실은 focal loss + dice loss. geometric prompt 혼합으로 학습.

**Data engine**

1. assisted-manual: 사람이 SAM으로 점 찍으면서 마스크. public 데이터로 시작해서 6번 재학습. 마스크당 34초 → 14초. 이미지당 20 → 44개. 12만 장, 430만 마스크.
2. semi-automatic: 자신 있는 마스크는 자동으로 깔고, 사람은 남은 물체만. 다양성을 늘리려고. 18만 장 추가로 총 1020만 마스크. 어려운 물체라 마스크당 다시 34초. 이미지당 72개.
3. fully automatic: 32x32 그리드 점을 넣고 점마다 여러 마스크. 자신 있는 것 + stable한 것만 남기고 NMS. 작은 마스크는 crop을 여러 장 본다. 1100만 장 전체에 적용해서 11억 마스크. SA-1B는 이 자동 마스크만 넣는다 (99.1%).

**SA-1B**

이미지 평균 해상도 3300x4950, 공개본은 짧은 변 1500. 얼굴이랑 번호판은 블러. 자동 vs 사람이 고친 마스크 IoU가 94%에서 90 초과. 기존 데이터보다 코너 커버가 넓고, 작/중간 마스크 비율이 높다. 지리적으로는 유럽·아시아 비중이 크고, 아프리카·저소득 국가는 여전히 적다. 그래도 아프리카만 마스크 2800만 개라 이전 데이터셋 전체보다 많다.

---

## 실험 구성과 결과는? (Experiments)

기본 모델은 ViT-H, 학습 데이터는 자동 마스크만.

**한 점 → 마스크 (23개 데이터셋)**

수중, ego-centric, 세포, 예술 등 학습에 없는 분포. 한 점은 애매해서 mIoU만 보면 불안정하다. 사람 평가(1~10)를 같이 본다.

한 점 mIoU에서 SAM이 RITM보다 23개 중 16개에서 이김. 3개 마스크 중 GT에 제일 맞는 걸 고르는 oracle이면 전부 이김. 사람 평가는 SAM이 항상 더 높고, 평균 7~9 (물체는 알아보겠고 작은 실수만). 점을 늘리면 격차는 줄어든다. SAM은 초고 IoU interactive용이 아니다.

**Edge detection (BSDS500)**

그리드 점으로 마스크를 많이 뽑고 Sobel. ODS 0.768. HED(0.788)보다는 낮고, 학습 안 한 zero-shot 옛 방법보다는 높다. recall은 높은데 precision이 낮다. 애노테이션에 없는 타당한 엣지도 많이 그린다.

**Object proposal (LVIS)**

자동 마스크를 proposal로. AR@1000이 ViTDet-H 63.0, SAM 59.3. medium/large, rare/common에서는 SAM이 앞선다. small이랑 frequent는 LVIS에 맞춘 detector가 이긴다. 마스크 1개만 내면 54.9로 떨어진다.

**Instance segmentation**

ViTDet 박스를 SAM에 넣음. mask AP는 COCO 51.0 vs 46.5, LVIS 46.6 vs 44.7로 detector가 앞선다. 사람 평가는 SAM이 더 높다 (8.1 vs 7.9). COCO GT 자체 평점이 7.6이라, detector가 데이터셋 버릇(구멍 없음, 단순 폴리곤)을 학습한 것으로 본다.

**Text-to-mask**

텍스트 라벨 없이, 마스크의 CLIP 이미지 임베딩으로 학습하고 테스트 때는 CLIP 텍스트 임베딩을 쓴다. 정성적으로는 `"a wheel"`, `"beaver tooth grille"` 같은 게 된다. 실패하면 점을 하나 더 주면 고쳐지는 경우가 있다. 저자들도 proof-of-concept라고 한다.

**Ablation**

data engine 단계가 쌓일수록 23-dataset mIoU가 오른다. 자동만 써도 전부 쓸 때보다 약 0.5 낮다. 이미지 11M vs 1M은 비슷하고, 0.1M은 많이 떨어진다. ViT-H는 ViT-B보다 분명 낫고 ViT-L보다는 조금 낫다. 더 키우는 이득은 거의 없다.

---

## 무엇을 발견했고 한계점은 무엇인가? (Discussion)

세그멘테이션도 프롬프트 가능한 인터페이스로 만들면, 학습 때 정의 안 한 과제를 시스템에 끼워 넣을 수 있다. CLIP이 DALL·E 부품이 되듯 SAM은 박스 detector, gaze, 3D 재구성 모듈과 붙일 수 있다. foundation model 정의랑은 맞지만, 능력의 대부분은 MAE가 아니라 대규모 지도학습에서 온다. 주석을 data engine으로 키울 수 있으면 supervised가 유효하다.

한계:

- 가는 구조는 놓치고, 작은 덩어리를 환각하고, zoom-in 하는 무거운 방법보다 경계가 덜 날카롭다.
- 점을 많이 주면 전용 interactive 모델이 더 잘할 수 있다.
- 이미지 인코더가 무거워서 end-to-end real-time은 아니다. 프롬프트 쪽만 50ms다.
- text-to-mask는 아직 약하다.
- semantic / panoptic을 간단한 프롬프트로 어떻게 할지는 분명하지 않다.
- 세포 분석처럼 도메인 전용 툴이 그 도메인에서는 앞설 수 있다.
- 사람 옷 세그멘테이션에서는 perceived gender 쪽 편향 조짐이 있다. SAM을 큰 시스템의 부품으로 쓰면 편향이 생길 수 있다.

---

## 결론 및 주요 요약은? (Conclusion)

세그멘테이션을 foundation model 시대로 올려 보려는 시도다. 기여는 promptable segmentation, SAM, SA-1B 세 개다. SAM이 실제로 foundation model이 될지는 쓰임에 달렸고, 마스크 10억 개랑 promptable 모델 공개가 다음 길을 연다고 본다.

---

## 기존 연구 대비 차별점은 무엇인가?

고정 클래스 semantic/instance 모델은 새 물체 정의가 어렵다. interactive 모델은 사람 클릭을 전제로 하고, 애매한 한 점에 여러 답을 안 낸다. CLIP은 분류/검색 인터페이스고 SAM은 마스크 인터페이스다. 데이터는 웹에서 긁는 대신 모델-in-the-loop로 마스크를 만든다.

---

## 핵심 아이디어는?

“이 이미지에서 이걸 잘라라”를 프롬프트로 받는 과제로 사전학습하면, 테스트 때 프롬프트만 바꿔서 여러 세그멘테이션 과제에 붙일 수 있다. 이미지 임베딩은 한 번만 계산하고, 애매하면 마스크를 여러 개 낸다. 마스크 데이터가 없으니 그 모델로 데이터를 또 만든다.

---

## 주요 수식과 코드 분석

논문에 GAN/CLIP처럼 닫힌 식 하나가 있는 건 아니다. 애매함을 다루는 학습이 핵심이다. 마스크 3개에 대해

$$
\mathcal{L} = \min_i \big( \mathcal{L}_{\mathrm{focal}}(\hat{m}_i, m) + \mathcal{L}_{\mathrm{dice}}(\hat{m}_i, m) \big)
$$

제일 잘 맞는 하나만 학습한다. 안 그러면 셔츠랑 사람을 평균한 이상한 마스크가 나온다. 추론 때는 예측 IoU가 높은 마스크를 고른다. automatic 단계에서는 threshold를 0.5-δ, 0.5+δ로 바꿔도 비슷한 마스크만 stable로 남긴다.

`code/sam.py`에서 min-loss랑 IoU ranking만 작게 돌려 봤다.

실행:

```
python papers/code/sam.py
```
