# [2021] LoRA: Low-Rank Adaptation of Large Language Models

- 저자: Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft)
- 링크: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- 코드: [github.com/microsoft/LoRA](https://github.com/microsoft/LoRA)

---

## 한 줄로

큰 모델은 얼려 두고, 가중치 업데이트 ΔW만 작은 행렬 두 개(B, A)의 곱으로 학습하는 방법

지금 PEFT의 기본기
학습 파라미터를 GPT-3 기준으로 1/10,000까지 줄이면서 full fine-tuning이랑 비슷하거나 더 잘 나온다

---

## 무슨 문제를 풀려고 했는가

pretrain 한 다음 downstream에 맞추는 게 NLP의 기본 흐름이다
근데 모델이 커지면 full fine-tuning이 부담이 된다

GPT-3 175B를 태스크마다 통째로 저장하면
태스크 하나당 175B짜리 복사본이 생긴다
배포도 어렵고, Adam optimizer state까지 들면 학습 VRAM도 1.2TB 근처까지 간다

이미 있던 대안도 각자 구멍이 있다

- **Adapter**: 레이어를 직렬로 끼워서 inference latency가 생긴다
  파라미터가 적어도, GPU는 병렬이라 작은 레이어가 순차로 돌면 느려진다
  GPT-2 medium, batch 1에서 AdapterH는 +30% 정도 느려짐
- **Prefix / prompt tuning**: 시퀀스 일부를 특수 토큰이 먹어서 실제 입력 길이가 줄어든다
  학습도 잘 안 되고, 토큰을 늘린다고 성능이 단조증가하지 않는다

그래서 목표는 이거 세 개다

1. 학습 파라미터를 확 줄인다
2. inference 때 속도가 full FT랑 같게 만든다
3. 품질은 FT를 따라간다

---

## 관련 연구 맥락 
1. **Full fine-tuning**: 전부 업데이트 / 제일 잘 나오지만 비싸다
2. **Adapter** (Houlsby, Lin, Pfeiffer): bottleneck MLP를 레이어 사이에 삽입 / 직렬이라 latency
3. **BitFit**: bias만 학습
4. **Prefix-tuning / prompt-tuning**: 입력 쪽에 학습 가능한 토큰을 붙인다
5. **Intrinsic dimension** (Aghajanyan et al., Li et al.): over-parametrized 모델이 실제로는 낮은 차원에서 학습된다는 관찰

LoRA는 5번의 관찰을 **가중치 업데이트**에 가져온 논문이다
모델 자체가 low-rank라는 말이 아니라, adaptation 때 바뀌는 ΔW가 low-rank일 거라는 가설

기존 low-rank 연구는 학습 처음부터 가중치를 분해하는 쪽이 많았다
LoRA는 pretrained W를 그대로 두고, 업데이트만 분해한다

---

## 핵심 아이디어

가설: downstream에 맞출 때 가중치 변화 ΔW의 intrinsic rank가 낮다

그래서

```
W = W₀ + ΔW
ΔW = BA     (B: d×r, A: r×k, r ≪ min(d, k))
W₀는 freeze, A랑 B만 학습
```

forward는 그냥 더하면 된다

```
h = W₀x + BAx
```

배포할 때는 `W ← W₀ + BA`로 합쳐 버리면
원래 모델이랑 구조가 똑같아서 **inference latency가 0**이다
태스크를 바꿀 때는 BA만 빼고 다른 B′A′를 더하면 된다

> [그림 넣을 곳] 논문 Figure 1 — frozen W 옆에 A, B를 병렬로 붙인 reparametrization
> A는 Gaussian 초기화, B는 0이라서 학습 시작 때 ΔW = 0

---

## 모델 구조

### 어디다 붙이나

이론상 dense layer면 다 된다
실험은 Transformer attention만 건드렸다

- 후보: `Wq, Wk, Wv, Wo` + MLP 두 개
- 기본 설정: **Wq, Wv만** LoRA / MLP는 freeze
- 파라미터 수: `|Θ| = 2 × L_LoRA × d_model × r`

파라미터 예산 18M으로 GPT-3에서 어디를 건드릴지 비교하면
`Wq`나 `Wk`만 하는 것보다 **Wq + Wv**가 제일 낫다
rank를 키워서 한 종류만 크게 학습하는 것보다, rank를 낮춰도 여러 행렬에 나눠 거는 쪽이 이득

### 초기화 / 스케일

- A ~ N(0, σ²)
- B = 0 → 시작 때 pretrained랑 동일
- `ΔWx`에 `α / r`을 곱한다
- α는 처음 시도한 r에 맞춰 고정 / r을 바꿔도 lr을 다시 안 만져도 되게

r을 d_model까지 올리면 (bias도 같이 학습할 때) full FT의 expressiveness에 가까워진다
adapter는 아무리 키워도 MLP가 되고, prefix는 긴 입력을 못 받는 쪽으로 수렴한다
LoRA만 r을 키우면 원래 모델에 가까워진다

---

## 주요 수식

full FT의 목표

```
max_Φ  Σ_{(x,y)} Σ_t log P_Φ(y_t | x, y_<t)
```

문제는 태스크마다 `|ΔΦ| = |Φ₀|`이라는 점이다

LoRA는 ΔΦ를 훨씬 작은 Θ로 인코딩한다

```
ΔΦ = ΔΦ(Θ),   |Θ| ≪ |Φ₀|
h = W₀x + BAx
```

GPT-3 175B에서 |Θ|는 |Φ₀|의 0.01%까지 줄어든다

**subspace similarity** (r이 커도 의미 있는 방향이 늘지 않는지 볼 때)

```
φ(A_{r=8}, A_{r=64}, i, j) = ||U^i_{r=8}ᵀ U^j_{r=64}||²_F  /  min(i, j)
```

0이면 완전 다른 공간, 1이면 같은 공간
r=8의 top 방향이 r=64에 거의 들어가 있다
나머지는 학습 중 쌓인 노이즈에 가깝다 → ΔW의 intrinsic rank가 진짜 낮다

---

## 코드로 보면

공식 구현은 `microsoft/LoRA`
지금은 HuggingFace PEFT가 사실상 표준이다
핵심만 적으면 이렇다

```python
# W₀는 freeze
# A: (r, k), B: (d, r)
h = W0(x) + (alpha / r) * B(A(x))

# 배포 때 merge
W.data += (B @ A) * (alpha / r)
# 태스크 전환
W.data -= (B @ A) * (alpha / r)
W.data += (B2 @ A2) * (alpha / r)
```

알아둘 점:

- adapter는 레이어 **뒤에 직렬**, LoRA는 같은 입력에 **병렬**
- merge하면 forward 그래프가 원래 모델이랑 같다
- 학습 때 optimizer state는 A, B만 들고 가면 된다
  GPT-3에서 VRAM 1.2TB → 350GB, 체크포인트 350GB → 35MB (r=4, Wq/Wv)

---

## 실험

NVIDIA V100
RoBERTa / DeBERTa는 GLUE, GPT-2는 E2E 등 NLG, GPT-3 175B는 WikiSQL / MNLI / SAMSum

### GLUE (NLU)

| 모델 | 방법 | 학습 파라미터 | Avg |
|---|---|---:|---:|
| RoBERTa-base | FT | 125M | 86.4 |
| RoBERTa-base | LoRA | 0.3M | **87.2** |
| RoBERTa-large | FT | 355M | 88.9 |
| RoBERTa-large | LoRA | 0.8M | **89.0** |
| DeBERTa-XXL | FT | 1500M | 91.1 |
| DeBERTa-XXL | LoRA | 4.7M | **91.3** |

파라미터를 수백 배 줄여도 FT를 따라잡거나 조금 이긴다

### GPT-2 E2E NLG

GPT-2 medium 기준 LoRA 0.35M이 FT 355M보다 BLEU가 높다 (70.4 vs 68.2)
prefix-layer(0.35M)도 이기면서, adapter가 가진 latency는 없다

### GPT-3 175B

| 방법 | 학습 파라미터 | WikiSQL | MNLI-m | SAMSum R1 |
|---|---:|---:|---:|---:|
| FT | 175B | 73.8 | 89.5 | 52.0 |
| BitFit | 14.2M | 71.3 | 91.0 | 51.3 |
| PreEmbed | 3.2M | 63.1 | 88.6 | 48.3 |
| AdapterH | 40.1M | 73.2 | 91.5 | 53.2 |
| LoRA | 4.7M | 73.4 | **91.7** | **53.8** |
| LoRA | 37.7M | **74.0** | 91.6 | 53.4 |

4.7M만으로 FT를 따라잡고, 어떤 태스크는 더 잘한다
prefix는 토큰을 너무 늘리면 성능이 떨어진다 (입력 분포가 pretrain이랑 멀어져서)
LoRA는 파라미터를 늘려도 성능이 안정적이다

> [그림 넣을 곳] 논문 Figure 2 — 학습 파라미터 수 vs WikiSQL / MNLI 정확도
> LoRA가 스케일도 좋고 점수도 높다

few-shot만으로는 부족하다
GPT-3 MNLI-m: few-shot 40.6 vs FT 89.5
데이터 몇 천 개만 있어도 파라미터를 조금은 건드리는 편이 이득

low-data (MNLI 100/1k/10k)에서도 LoRA가 prefix보다 훨씬 낫고, 100개에선 FT보다도 높다 (63.8 vs 60.2)

### 왜 r이 작아도 되나

GPT-3에서 Wq+Wv면 **r=1**로도 이미 충분하다
WikiSQL 73.4, MNLI 91.3 근처

다른 r끼리, 다른 seed끼리 subspace를 겹쳐 보면
쓸모 있는 방향은 몇 개 안 되고 겹친다
r을 키워도 그 공간을 더 의미 있게 넓히진 않는다

다만 모든 태스크가 그런 건 아니다
논문도 말한다 — downstream이 아예 다른 언어면 r을 작게 잡으면 질 수 있다

### ΔW는 W랑 무슨 관계인가

ΔW는 W의 top singular 방향을 그대로 반복하지 않는다
이미 W에 있지만 **강조되지 않았던** 방향을 키운다
r=4에서 증폭 배율이 약 21.5배 (`||ΔW|| / ||UᵀWVᵀ||`)

> [그림 넣을 곳] 논문 Figure 3, 4 — r=8 vs r=64, seed 간 subspace similarity
> top 1방향만 많이 겹침

해석하면
pretrain이 여러 특징을 이미 갖고 있고
LoRA는 그 태스크에 필요한 축만 크게 키운다

---

## 기존 연구 대비 뭐가 다른가

- Adapter: 구조가 bottleneck으로 비슷해 보이지만 **직렬 vs 병렬**
  LoRA는 merge가 되어서 latency가 없다
- Prefix-tuning: 시퀀스 길이를 안 깎는다
- BitFit: bias만 vs 행렬 업데이트를 low-rank로
- Full FT: 태스크마다 모델 통째 저장 vs 베이스 하나 + 작은 A,B 여러 개
- 기존 low-rank 학습: 원 모델을 분해 vs **freeze된 모델의 업데이트만** 분해
- COMPACTER 등: Kronecker로 adapter를 더 압축 / LoRA랑 같이 쓸 여지는 남김
- prefix랑 결합도 가능 (Appendix E)
  LoRA+PrefixEmbed는 WikiSQL에서 둘 다보다 좋다 → 어느 정도 orthogonal

---

## 한계 / 아쉬운 점

- merge해 두면 태스크가 다른 샘플을 한 배치에 넣기 어렵다
  latency가 덜 중요하면 merge 안 하고 샘플마다 A,B를 고르면 된다
- 어느 행렬에 붙일지는 휴리스틱 (attention만, 그중에서도 Wq/Wv)
  MLP / LayerNorm / bias는 후속으로 남김
- r이 항상 1~4면 되는 건 아님
  태스크가 pretrain이랑 많이 다르면 rank가 더 필요할 수 있다
- GPT-2 medium은 GPT-3보다 r이 조금 더 필요 (E2E에서 BLEU는 r=4, val loss는 r=16 근처)
- 왜 fine-tuning이 되는지에 대한 설명은 아직 관찰 수준
  다만 full FT보다 ΔW를 보기 쉬운 도구이긴 하다

---

## 결론만

W는 그대로 두고 ΔW = BA만 학습하면
저장·학습 비용이 크게 줄고, 합치면 속도는 FT랑 같다
품질은 RoBERTa부터 GPT-3까지 FT를 따라가거나 이긴다

남는 메시지 세 개

1. adaptation의 변화량은 생각보다 낮은 rank로 충분하다
2. 그 변화는 W의 주성분이 아니라, 이미 있지만 약했던 방향을 키우는 쪽에 가깝다
3. 그래서 큰 모델 하나를 공유하고 태스크는 작은 모듈로 갈아끼울 수 있다

지금 LoRA를 쓰는 이유랑 거의 같다
베이스는 하나, 어댑터만 여러 개, 서빙 때 merge
이 논문이 그 레시피를 처음 제대로 정리해 준 셈이다
