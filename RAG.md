# [2021 NeurIPS] RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

- 저자: Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (FAIR / UCL / NYU)
- 링크: [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- 코드: HuggingFace Transformers `examples/rag`

---

## 한 줄로

파라미터 안에 지식을 넣는 LM(BART)에, 위키피디아를 뽑아 오는 retriever(DPR)를 붙여서, 지식 의존 NLP를 generation으로 푸는 논문

지금 쓰는 RAG의 원조
검색한 문서를 latent variable로 두고 top-K에 대해 marginalize하는 게 핵심

---

## 무슨 문제를 풀려고 했는가

당시 pretrained LM은 파라미터에 factual knowledge를 어느 정도 저장한다
근데 한계가 뚜렷하다

- 지식을 정확히 꺼내 쓰기가 어렵다
- 틀린 말을 잘한다 (hallucination)
- 왜 그렇게 답했는지 근거를 못 댐
- 세상이 바뀌면 모델을 다시 학습해야 한다
- knowledge-intensive task에서는 task-specific 모델보다 약하다

REALM, ORQA 같은 hybrid 모델이 나오긴 했는데, extractive QA에만 쓰였다
이 논문은 그걸 seq2seq generation까지 끌어올렸다

> 여기서 knowledge-intensive task는, 사람이 외부 지식 없이 풀기 어려운 문제
> ODQA, fact verification, 사실 기반 생성 같은 것들

---

## 관련 연구 맥락

ODQA가 이 논문의 직전 라인

1. **DrQA**: 위키에서 문서 찾고, reader가 span을 뽑아 답
2. **ORQA**: retriever도 같이 학습 / bi-encoder로 질문과 비슷한 passage top-K를 뽑음
3. **REALM**: retriever-reader를 end-to-end로 pretrain / salient span masking 사용
4. **DPR**: dense retriever / 이 논문 retriever의 초기값
5. **T5 / BART**: retrieval 없이 파라미터만으로 답을 생성하는 closed-book QA

선행 연구들은 태스크마다 retrieval을 따로 붙이거나, extractive reader에만 hybrid를 썼다
RAG는 **하나의 seq2seq 구조**로 QA, 생성, 분류까지 같이 가져가려 했다

memory network랑도 비슷한데, 메모리가 embedding이 아니라 **raw text**
사람이 읽을 수 있고, 인덱스만 갈아끼우면 지식 업데이트가 된다

---

## 핵심 아이디어

```
입력 x
  → Query Encoder가 q(x) 만듦
  → Wikipedia index에서 MIPS로 top-K 문서 z 검색
  → BART가 (x + z)를 보고 y 생성
  → 문서 z를 latent로 보고 확률을 더함 (marginalize)
```

두 종류가 있다

| | RAG-Sequence | RAG-Token |
|---|---|---|
| 문서 사용 | 한 문서로 문장 전체를 생성 | 토큰마다 다른 문서를 볼 수 있음 |
| 잘 맞는 곳 | 짧은 답 (ODQA), 문장 단위 생성 | 여러 문서를 섞어야 하는 생성 (Jeopardy) |
| decoding | 문서별로 beam search 후 합침 | 일반 beam search에 그냥 넣음 |

분류 태스크는 target 길이가 1이라 둘이 같다

> [그림 넣을 곳] 논문 Figure 1 — Retriever + Generator + Marginalize 전체 구조

---

## 모델 구조

### Retriever: DPR

BERT bi-encoder

- 문서 인코더: `d(z) = BERT_d(z)` → 미리 다 뽑아 두고 인덱스로 저장
- 쿼리 인코더: `q(x) = BERT_q(x)` → 학습 때 같이 업데이트
- 점수: 내적 `d(z)ᵀ q(x)`
- 검색: FAISS MIPS (HNSW)

학습 중 document encoder랑 인덱스는 **고정**
REALM처럼 인덱스를 계속 다시 만드는 건 비싸서 안 한다
query encoder랑 BART만 학습해도 성능이 잘 나온다

### Generator: BART-large

400M seq2seq
입력은 그냥 `x`랑 가져온 `z`를 concatenate해서 넣는다
별도 fusion 레이어는 없다

### 학습

문서가 정답인지에 대한 직접 supervision은 없다
`(x, y)` 쌍만 보고

```
Σ_j -log p(y_j | x_j)
```

를 줄인다
정답 생성에 도움 되는 문서를 retriever가 알아서 더 잘 가져오게 된다

---

## 주요 수식

검색 확률:

```
p_η(z|x) ∝ exp( d(z)ᵀ q(x) )
```

**RAG-Sequence** — 문서 하나 고르고, 그 문서로 전체 y를 생성한 뒤 합산

```
p(y|x) ≈ Σ_{z ∈ top-K} p_η(z|x)  Π_i p_θ(y_i | x, z, y_{<i})
```

sum이 바깥, product가 안쪽
문서가 sequence 전체를 책임진다

**RAG-Token** — 토큰마다 문서를 다시 고른다

```
p(y|x) ≈ Π_i  Σ_{z ∈ top-K} p_η(z|x) p_θ(y_i | x, z, y_{<i})
```

product가 바깥, sum이 안쪽
토큰 단위로 문서를 갈아탈 수 있다

둘 차이를 한 줄로 보면
Sequence는 “이 문서가 이 답 전체의 근거”
Token은 “이 단어는 저 문서, 저 단어는 이 문서”

### Decoding

- **Token**: 매 step에서 top-K 문서 확률을 합친 뒤 일반 beam search
- **Sequence**: 문서마다 따로 beam search
  - Thorough: 다른 문서 beam에 없던 후보도 다시 forward해서 확률을 더함
  - Fast: beam에 안 나온 후보는 확률 0으로 침 / 긴 생성에 사용

QA는 답이 짧아서 Thorough + greedy
MS-MARCO / Jeopardy는 Fast + beam 4

---

## 코드로 보면

실제 구현은 HuggingFace RAG
흐름만 적으면 대략 이런 식

```python
# 1) query encode + retrieve
q = query_encoder(x)                  # BERT_q
docs, doc_scores = faiss.search(q, K) # p_η(z|x)

# 2) 각 문서를 context로 붙여서 generate
#    input: [query] [SEP] [passage]
for z, score in zip(docs, doc_scores):
    logits = bart(concat(x, z))       # p_θ(y | x, z)

# 3) marginalize
# RAG-Sequence: 문서별 sequence logprob에 score를 더하고 logsumexp
# RAG-Token:    토큰 vocab 분포를 문서 score로 가중합
```

알아둘 점:

- document encoder는 인덱스 빌드 때만 쓴다 / forward 때 안 굴린다
- 학습 그래프는 `query encoder → retrieval score → generator NLL` 쪽으로만 흐름
- top-K truncation이라 진짜 전체 Wikipedia에 대한 합은 아니다 / 근사

---

## 실험

지식 소스는 전부 **Wikipedia 2018.12**
문서를 100단어 chunk로 잘라 21M개
학습 때 K는 5 또는 10

### Open-domain QA (EM)

| Model | NQ | TriviaQA | WebQ | CuratedTrec |
|---|---:|---:|---:|---:|
| T5-11B (closed-book) | 34.5 | — | 37.4 | — |
| REALM | 40.4 | — | 40.7 | 46.8 |
| DPR | 41.5 | 57.9 | 41.1 | 50.6 |
| RAG-Token | 44.1 | 55.2 | **45.5** | 50.0 |
| RAG-Sequence | **44.5** | 56.8 | 45.2 | **52.2** |

- NQ / WebQ / CuratedTrec에서 SOTA
- re-ranker, extractive reader 없이도 DPR을 이김
- 가져온 문서에 정답이 없어도 NQ에서 11.8%는 맞춤 / extractive면 0
- T5-11B(110억)보다 파라미터는 훨씬 적은데(학습 가능 6.26억) 점수는 더 높다

생성으로 푸는 이유가 여기 있다
문서에 답이 그대로 없어도, 힌트만 있으면 답을 만들어 낼 수 있다

### 생성 / 분류

**MS-MARCO** (gold passage 없이 open-domain으로 품)

- RAG-Seq가 BART보다 BLEU +2.6, ROUGE-L +2.6
- gold를 쓰는 SOTA보다는 낮지만, 위키만 보고 이 정도면 괜찮은 편

**Jeopardy 질문 생성**

- entity가 주어지면 그 사실을 묻는 Jeopardy 문장을 만드는 태스크
- RAG-Token이 제일 잘한다
  질문 하나에 사실이 두 개 들어가는 경우가 많아서, 문서를 토큰마다 갈아타는 쪽이 유리
- 사람 평가 (BART vs RAG, 452쌍)
  - factuality: RAG가 낫다 42.7% / BART가 낫다 7.1%
  - specificity: 37.4% vs 16.8%

> [그림 넣을 곳] 논문 Figure 2 — Hemingway 예시
> "The Sun Also Rises" 생성 때는 doc2, "A Farewell to Arms" 때는 doc1 posterior가 올라감
> 제목 첫 토큰 이후엔 posterior가 평평해짐 → 나머지는 BART 파라미터에 이미 들어 있다

이게 이 논문에서 제일 재미있는 관찰
non-parametric이 단서를 던져 주고, parametric이 제목을 완성한다
두 메모리가 역할을 나눈다

**FEVER** (fact verification)

- retrieval supervision 없이 claim만 넣고 supports / refutes / NEI를 분류
- 3-way: SOTA 대비 4.3%p 낮음 / 근데 SOTA는 파이프라인 + evidence 정답 학습
- top-1 문서가 gold article인 비율 71%, top-10이면 90%

### ablation에서 남는 것

- retriever를 freeze하면 전부 떨어진다 → retrieval을 태스크에 맞게 배우는 게 먹힌다
- BM25로 바꾸면 QA는 크게 하락 / FEVER만 BM25가 더 좋다 (claim이 entity 중심이라 lexical overlap이 잘 맞음)
- 인덱스 교체 실험: 2016/2018 위키를 바꿔 끼우면 그 시점 국가원수 질문에 ~70%로 맞춤
  인덱스를 엇갈리게 넣으면 4~12%
  **재학습 없이 지식 업데이트가 된다**
- 생성 diversity(distinct trigram)도 RAG > BART / 별도 diversity decoding을 안 써도 된다

> [그림 넣을 곳] 논문 Figure 3 — K를 늘리면 RAG-Sequence QA는 계속 오르고, RAG-Token은 K=10 근처에서 꺾임

---

## 기존 연구 대비 뭐가 다른가

- REALM / ORQA: extractive → RAG는 **generation** / span에 없는 답도 생성 가능
- DPR: retriever + extractive reader + re-ranker → RAG는 reader/re-ranker 없이 seq2seq 하나로 처리
- T5 closed-book: 지식 업데이트가 재학습 → RAG는 인덱스 교체
- 태스크별 retrieve-and-extract 파이프라인 → RAG는 같은 레시피로 QA / NLG / 분류
- memory network: 메모리가 vector가 아니라 텍스트 → 해석이 되고 수정이 됨
- retrieve-and-edit: 학습 pair를 가져와 고치는 쪽 → RAG는 evidence 문서를 가져와 여러 개를 합침

---

## 한계 / 아쉬운 점

논문이 따로 Limitations 섹션을 두진 않는다
본문이랑 appendix에 흩어져 있다

- **Retrieval collapse**: story generation처럼 사실이 덜 중요한 태스크에선 retriever가 입력이 달라도 같은 문서만 가져온다 → generator가 문서를 무시하고 BART랑 같아진다
- document encoder를 안 풀어서, 인덱스가 태스크에 맞게 바뀌진 않는다
- FEVER evidence sentence 추출은 안 한다 / dump가 달라서
- Wikipedia 밖의 질문(날씨 등)은 parametric에 의존한다 / null document를 넣어 봤지만 이득이 없어서 뺐다
- 지식 소스가 위키라서, 위키의 편향/오류를 그대로 가져온다
- 학습 가능 파라미터는 6.26억이지만, 인덱스 벡터가 21M × 768이라 메모리는 따로 필요하다 (압축하면 CPU 36GB)

---

## 결론만

parametric(BART) + non-parametric(위키 인덱스)를 seq2seq에 붙이면, 지식 의존 태스크를 하나의 fine-tuning 레시피로 풀 수 있다

남는 메시지는 세 개

1. 생성 + retrieval이 extractive보다 유연하다
2. 두 메모리가 실제로 역할을 나눈다 (Jeopardy 예시)
3. 인덱스만 바꾸면 지식을 업데이트할 수 있다

지금 RAG 시스템(검색 → prompt에 붙이기 → LLM 생성)이랑 모양이 비슷해 보이는데, 이 논문은 검색 점수를 generator likelihood랑 **같이 미분**해서 retriever까지 태스크에 맞게 학습한다
그 지점이 요즘 pipeline RAG랑 제일 다른 부분
