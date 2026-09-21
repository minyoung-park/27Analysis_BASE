# Learning Transferable Visual Models From Natural Language Supervision

Alec Radford, Jong Wook Kim 외 (OpenAI, 2021). arXiv:2103.00020.

보통 CLIP이라고 부르는 논문. Contrastive Language-Image Pre-training.

---

## 어떤 문제를 해결하려 했는가? (Abstract)

지금 컴퓨터 비전 모델은 미리 정해 둔 카테고리를 맞추게 학습한다. 새 개념을 넣으려면 라벨 데이터를 또 모아야 해서 일반성이 떨어진다.

이미지에 붙은 원문 텍스트에서 바로 배우면 감독 신호가 훨씬 넓다. 이 논문은 캡션이 어떤 이미지랑 짝인지 맞추는 사전학습이, 인터넷에서 모은 4억 (이미지, 텍스트) 쌍에서 SOTA급 이미지 표현을 처음부터 배우는 데 효율적이라고 한다.

사전학습 뒤에는 자연어로 시각 개념을 가리켜서 downstream에 zero-shot으로 옮긴다. OCR, 비디오 행동 인식, geo-localization, fine-grained 분류 등 30개 넘는 데이터셋에서 본다. 데이터셋별 학습 없이도 완전 지도학습 베이스라인과 경쟁하는 경우가 많다. ImageNet에서는 학습셋 128만 장을 안 쓰고도 원조 ResNet-50 정확도에 맞춘다.

코드와 가중치: https://github.com/OpenAI/CLIP

---

## 연구 동기와 문제점은 무엇인가? (Introduction)

NLP는 raw text 사전학습이 바꿨다. GPT, BERT, T5처럼 task-agnostic 목적함수가 스케일되고, 텍스트를 입출력으로 통일하니까 zero-shot도 된다. GPT-3는 과제별 데이터가 거의 없어도 전용 모델이랑 비슷하다.

비전은 아직 ImageNet 같은 crowd-label 사전학습이 기본이다. 웹 텍스트로 비전에서도 비슷한 돌파가 되나?

오래된 선행은 있다. 캡션의 명사/형용사를 예측하거나(Mori et al.), 캡션 단어로 이미지 표현을 배우고(Joulin et al., YFCC100M), visual n-gram으로 zero-shot을 시도한 것도 있다(Li et al.). 최근엔 VirTex, ICMLM, ConVIRT가 transformer LM, MLM, contrastive로 이미지-텍스트를 학습했다.

근데 벤치마크 숫자가 낮다. Li et al. ImageNet zero-shot이 11.5%다. 당시 SOTA 88.4%, 고전 비전 50%보다도 낮다. 대신 hashtag 예측(Instagram)이나 JFT-300M 같은 좁은 약감독이 더 잘 먹혔다. 이쪽은 클래스 수가 1000이나 18291로 고정이고, softmax가 고정이라 zero-shot이 약하다.

스케일도 다르다. hashtag/JFT 모델은 GPU-year 단위고, VirTex류는 10~20만 장에 accelerator-day 수준이다. 이 논문은 그 간격을 메운다. 4억 쌍 WIT 데이터로 ConVIRT를 단순화한 CLIP을 scratch부터 학습한다. compute를 거의 100배 구간에서 8개 모델을 돌리면 transfer 성능이 compute의 매끈한 함수로 나온다. GPT처럼 사전학습 중에 OCR, 위치 추정, 행동 인식 같은 일을 같이 배운다.

---

## 관련 연구 동향은 어떠한가? (Related Works)

본문 관련연구는 Introduction이랑 8절에 나눠 있다.

자연어를 비전 감독으로 쓰는 건 이미지 검색(Mori et al. 1999)부터 있다. 이후 CCA, ranking으로 joint embedding을 배웠고, deep 시대에 성능이 올랐다. 캡션 데이터는 Pascal1K, Flickr8K/30K가 작고, Conceptual Captions 등은 필터가 세서 아직 100만~1000만 규모다. CLIP의 WIT는 4억이다.

webly supervised learning은 검색 쿼리를 라벨로 쓴다. CLIP은 쿼리가 아니라 이미지와 같이 나온 전체 텍스트를 쓴다.

VirTex는 caption 생성, ICMLM은 masked LM, ConVIRT는 의료 영상에서 contrastive. CLIP은 ConVIRT를 스케일하고 단순화한 쪽에 가깝다.

Instagram hashtag, JFT-300M은 약감독이지만 클래스 집합이 닫혀 있다.

최근 VQA용 vision-language 모델(ViLBERT 등)은 detector + BERT를 붙여 복잡하다. CLIP은 이미지 인코더와 텍스트 인코더 두 개, 목적함수는 pairing 하나다.

NLP 쪽 GPT-1/2 zero-shot이 CLIP의 task-learning 평가 방식의 모델이다. Visual N-Grams가 비전에서 비슷한 zero-shot 평가를 먼저 했다.

---

## 연구 접근법과 모델 구조는 어떻게 되는가? (Method)

**데이터 (WIT)**

MS-COCO, Visual Genome은 각 10만 장 수준. YFCC100M은 메타데이터가 지저분해서 영어 자연어만 남기면 1500만 장, ImageNet 정도다. 그래서 웹에서 4억 쌍을 새로 모은다. Wikipedia에 자주 나오는 단어, PMI 높은 bi-gram, WordNet synset 등 50만 쿼리로 찾고, 쿼리당 최대 2만 쌍으로 클래스 밸런스를 맞춘다. 총 단어 수는 GPT-2 WebText랑 비슷하다.

**왜 contrastive인가**

처음엔 이미지 CNN + 텍스트 transformer로 캡션을 생성하게 했다. 파라미 6300만 transformer가 ResNet-50보다 compute를 두 배 쓰는데, ImageNet zero-shot은 bag-of-words 예측보다 3배 느렸다. 캡션 단어를 정확히 맞추는 일이 너무 어렵다.

그래서 배치 안에서 어떤 텍스트가 어떤 이미지랑 짝인지만 맞춘다. BoW 예측을 contrastive로 바꾸면 zero-shot ImageNet 학습 속도가 또 4배 빨라진다. 생성 모델보다 contrastive가 compute 대비 표현 학습이 잘 된다는 관찰이랑 같다.

배치 크기가 N이면 가능한 짝은 NxN개다. 맞는 N개 쌍의 cosine similarity는 키우고, 나머지 N^2-N개는 줄인다. similarity에 softmax를 걸어서 대칭 cross-entropy를 쓴다. N-pair loss / InfoNCE / ConVIRT랑 같은 계열이다.

ConVIRT보다 단순하다. ImageNet 가중치 초기화 없음. 비선형 projection 없음. 선형 projection만. 텍스트에서 문장 샘플링도 거의 안 함. augmentation은 resize 후 random square crop. temperature $\tau$는 log-parameter로 학습한다.

의사코드 핵심:

$$
I_e = \mathrm{normalize}(I_f W_i),\quad
T_e = \mathrm{normalize}(T_f W_t)
$$

$$
\mathrm{logits} = I_e T_e^\top \cdot e^{\tau}
$$

$$
\mathcal{L} = \frac{1}{2}\big(\mathrm{CE}(\mathrm{logits}, y) + \mathrm{CE}(\mathrm{logits}^\top, y)\big)
$$

정답 라벨 y는 (0, 1, ..., N-1). 대각선이 정답 짝이다.

**인코더**

이미지: ResNet-50 계열(ResNet-D, antialiased pooling, attention pooling)이랑 ViT. 텍스트: GPT-2 스타일 transformer, 12층, width 512, head 8, 약 6300만. BPE vocab 49152, max length 76. `[EOS]` hidden을 텍스트 특징으로 쓴다. masked self-attention.

스케일: ResNet은 width, depth, resolution을 같이 키움 (RN50, RN101, RN50x4, x16, x64). ViT는 B/32, B/16, L/14. 텍스트 인코더는 width만 이미지에 맞춰 키운다. 32 epoch, AdamW, cosine LR, batch 32768. $\tau$ 초기값 0.07에 해당하는 값, logit scale은 100으로 클립. 제일 큰 ResNet은 V100 592장으로 18일, 제일 큰 ViT는 256장으로 12일. 결과는 보통 ViT-L/14@336px.

**Zero-shot**

클래스 이름을 텍스트 인코더에 넣어서 분류기 가중치를 만든다. 이미지 임베딩이랑 cosine similarity, $\tau$로 스케일, softmax. 선형분류기인데 가중치를 텍스트가 생성하는 hypernetwork로 보면 된다. 프롬프트는 `"A photo of a {label}."`가 기본. 다의어(crane, boxer)나 문장 분포를 맞추려고. 데이터셋마다 `"a type of pet"`, `"satellite photo"`처럼 문맥을 붙이면 더 좋다. 여러 프롬프트 임베딩을 평균내는 앙상블도 쓴다. ImageNet에서 프롬프트+앙상블이 클래스 이름만 쓸 때보다 약 5%p.

---

## 실험 구성과 결과는? (Experiments)

Visual N-Grams 대비 ImageNet 11.5% → 76.2%. aYahoo 72.4 → 98.4, SUN 23.0 → 58.5. ImageNet top-5 95%로 Inception-V4랑 비슷하고, 원조 ResNet-50 zero-shot 매칭.

30개 넘는 데이터셋에서 task-specific 지도학습과 비교. 평균적으로 꽤 경쟁한다. 다만 차종, 꽃, 항공기 같은 fine-grained랑 물체 개수 세기, 차까지 거리처럼 사전학습에 없을 법한 추상 과제는 약하다.

같은 CLIP feature 위에서 보면 zero-shot이 4-shot logistic regression이랑 비슷하다. 한 장이 여러 개념을 담아서 example-based 학습이 애매한 반면, 텍스트는 개념을 바로 지정한다. 다른 모델 16-shot(BiT-M)이랑 평균이 비슷하다. 데이터셋마다 효율 차이가 크다. Flowers102, EuroSAT는 1-shot보다 못하고, ImageNet은 16-shot이랑 같다. median 5.4, mean 20.8 examples/class. fully supervised linear probe보다는 보통 10~25%p 낮다. 둘의 상관은 0.82.

linear probe로 보면 CLIP feature가 ImageNet 사전학습 모델보다 task shift에 덜 묶인다. 같은 ImageNet 점수에서 transfer가 더 높다.

robustness: ImageNet 모델은 자연 분포 이동에서 실수가 많이 는다. zero-shot CLIP은 in-distribution 대비 OOD 갭을 최대 75% 줄인다. CLIP feature로 ImageNet linear probe를 하면 ImageNet은 76.2 → 85.4로 오르는데, OOD 평균은 거의 안 오르거나 떨어진다. ImageNet 분포에 맞추면 그 분포 근처만 좋아진다.

데이터 오염: 35개 중 9개는 overlap 없음. median overlap 2.2%. 정확도가 overlap 때문에  inflating된 양은 거의 작고, 제일 큰 추정이 Birdsnap +0.6%.

---

## 무엇을 발견했고 한계점은 무엇인가? (Discussion)

캡션 생성보다 pairing contrastive가 스케일에 맞다. 자연어가 열린 개념 집합과 zero-shot 인터페이스를 같이 준다. zero-shot이 few-shot보다 나을 수 있는 이유는 개념을 말로 전달해서다. ImageNet에 적응하면 in-distribution만 오르고 robustness는 깎일 수 있다.

한계:

- zero-shot SOTA까지는 compute가 약 1000배 더 필요할 것 같고, 지금 하드웨어로는 비현실적이다.
- fine-grained, counting, 진짜로 본 적 없는 과제는 거의 랜덤에 가깝다.
- MNIST 손글씨는 88%라서 픽셀 logistic regression보다 못하다. 사전학습에 MNIST 비슷한 이미지가 거의 없다. 큰 데이터로 in-distribution을 넓혀서 brittle generalization을 우회하려는 쪽에 가깝다.
- 캡션처럼 새 문장을 생성하진 못한다. 주어진 클래스 집합 안에서만 고른다.
- 데이터 효율은 그대로다. 32 epoch면 이미지 128억 장을 본 셈이다.
- 개발 과정에서 validation 전체를 계속 봐서 진짜 zero-shot이라고 보기 어렵다.
- 웹 이미지-텍스트 그대로라 사회적 편향을 배운다. FairFace 프로브에서 Black 이미지를 비인간 클래스에 더 많이 넣고, 범죄 관련 클래스에도 성별/나이 격차가 있다. 클래스 설계에 따라 편향이 크게 바뀐다.
- few-shot을 직접 최적화하지 않아서, zero-shot에서 few-shot으로 가면 성능이 직관과 다르게 떨어질 수 있다. 사람은 1-shot에서 크게 오른다.

---

## 결론 및 주요 요약은? (Conclusion)

별도 Conclusion 절은 짧고, 6절 한계랑 7절 영향이 그 역할이다. 자연어 감독으로 비전 모델을 크게 학습하면, 고정 클래스 softmax 없이도 여러 데이터셋에 zero-shot으로 옮길 수 있다. pairing contrastive가 캡션 생성보다 효율적이고, 텍스트 인코더가 분류기를 그 자리에서 만든다. 다만 스케일, 추상 과제, 편향, few-shot 결합은 남아 있다.

---

## 기존 연구 대비 차별점은 무엇인가?

ImageNet/JFT 사전학습은 클래스 집합이 닫혀 있다. Visual N-Grams는 zero-shot 아이디어는 같지만 스케일과 목적함수가 약하다. VirTex는 단어를 생성하고, CLIP은 짝만 맞춘다. ConVIRT는 의료 도메인 contrastive고 CLIP은 웹 스케일 + 단순화(scratch, 선형 projection, 학습 가능한 $\tau$). VQA 멀티모달 모델보다 구조가 훨씬 얇다.

---

## 핵심 아이디어는?

이미지와 텍스트를 같은 공간에 넣고, 배치에서 맞는 짝의 cosine을 키운다. 그러면 이미지 표현이랑 언어가 연결된다. 테스트 때는 클래스 이름을 텍스트 인코더에 넣어서 분류기를 만든다. 라벨 공간이 자연어라서 새 과제도 프롬프트로 정의한다.

