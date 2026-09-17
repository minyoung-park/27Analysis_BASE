# Generative Adversarial Nets

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio (Université de Montréal, 2014)

arXiv:1406.2661v1. 흔히 GAN 논문이라고 부르는 원논문.

---

## 어떤 문제를 해결하려 했는가? (Abstract)

생성 모델을 적대적 학습으로 추정하는 프레임워크를 제안한다. 생성기 $G$는 데이터 분포를 따라 샘플을 만들고, 판별기 $D$는 그 샘플이 진짜 데이터인지 $G$가 만든 건지 구분한다. $G$의 목표는 $D$를 속이는 것이다. 두 모델이 minimax 게임을 한다.

함수 공간이 충분하면 해는 하나다. $G$가 데이터 분포를 복원하고 $D(x)=\frac{1}{2}$가 된다. $G$와 $D$를 MLP로 두면 전부 backpropagation으로 학습할 수 있다. 학습이든 샘플링이든 Markov chain이나 approximate inference가 필요 없다.

---

## 연구 동기와 문제점은 무엇인가? (Introduction)

딥러닝은 이미지, 음성, 텍스트 같은 데이터의 확률분포를 계층적으로 배우고 싶어 한다. 그런데 당시 성공은 거의 분류 모델 쪽이었다. backprop, dropout, ReLU 같은 piecewise linear unit이 잘 맞아서다.

생성 모델은 영향이 적었다. maximum likelihood를 하려면 intractable한 확률 계산을 근사해야 하고, 생성 쪽에서는 ReLU 같은 unit을 쓰기 어려웠다.

이 논문은 그 계산을 피하려고 한다. 생성기를 위조범, 판별기를 경찰로 본다. 위조범은 가짜를 진짜처럼 만들고, 경찰은 가짜를 잡는다. 경쟁하다 보면 가짜가 진짜와 구분이 안 되는 지점까지 간다.

이 글에서 다루는 특수 사례는 $G$와 $D$가 둘 다 MLP인 경우다. 이걸 adversarial nets라고 부른다. 학습은 backprop과 dropout만 쓰고, 샘플은 $G$에 노이즈를 넣어 한 번 forward하면 된다.

---

## 관련 연구 동향은 어떠한가? (Related Works)

RBM, DBM 같은 undirected 모델은 partition function 때문에 MCMC가 필요하고, mixing이 잘 안 된다. DBN은 undirected랑 directed가 섞여 있어서 둘의 계산 문제를 같이 가진다.

likelihood를 직접 안 쓰는 기준으로 score matching, NCE가 있다. 둘 다 정규화 상수만 모르는 density를 식으로 쓸 수 있어야 한다. DBN/DBM처럼 latent가 여러 층이면 그것도 어렵다. NCE는 생성 모델 자체가 고정된 노이즈 분포랑 진짜를 구분하는데, 모델이 어느 정도 맞기 시작하면 학습이 급격히 느려진다.

확률을 직접 정의하지 않고 샘플만 뽑는 기계도 있다. GSN은 denoising autoencoder를 확장한 것인데, 파라미터화된 Markov chain 한 스텝을 학습한다. GAN은 샘플링에 체인이 필요 없다. 생성 때 피드백 루프가 없어서 ReLU를 쓰기에도 낫다.

비슷한 시기에 VAE(auto-encoding variational Bayes), stochastic backpropagation도 나왔다. 그쪽은 추론 네트워크로 likelihood를 근사한다.

---

## 연구 접근법과 모델 구조는 어떻게 되는가? (Method)

노이즈 $z \sim p_z(z)$를 MLP $G(z;\theta_g)$에 넣어서 데이터 공간의 샘플을 만든다. 이게 $p_g$다. 판별기 $D(x;\theta_d)$는 스칼라 하나를 내고, $x$가 진짜 데이터에서 왔을 확률이라고 해석한다.

$D$는 진짜/가짜를 잘 맞추게 학습하고, $G$는 $\log(1-D(G(z)))$를 줄여서 $D$를 속인다. value function은

$$
\min_G \max_D V(D,G)
= \mathbb{E}_{x \sim p_{data}}[\log D(x)]
+ \mathbb{E}_{z \sim p_z}[\log(1-D(G(z)))]
$$

이론상 $D$를 안쪽 루프에서 끝까지 최적화하면 좋지만, 계산이 비싸고 오버핏된다. 그래서 $D$를 $k$번 업데이트하고 $G$를 1번 업데이트한다. 실험에서는 $k=1$을 썼다.

초반에 $G$가 너무 구리면 $D$가 가짜를 거의 확실히 거절한다. 그러면 $\log(1-D(G(z)))$가 saturate돼서 $G$ 그래디언트가 거의 안 나온다. 그래서 실제로는 $G$가 $\log D(G(z))$를 최대화하게 바꾼다. 고정점은 같고 초반 그래디언트만 살아난다.

학습 루프(Algorithm 1):

1. 노이즈 미니배치, 진짜 데이터 미니배치를 뽑는다.
2. $D$를 올려서 $\log D(x)+\log(1-D(G(z)))$를 키운다.
3. 노이즈를 다시 뽑고 $G$를 내려서 $\log(1-D(G(z)))$를 줄인다.

이론(비모수, 용량이 충분할 때):

- $G$가 고정이면 최적 판별기는 $D_G^*(x)=\dfrac{p_{data}(x)}{p_{data}(x)+p_g(x)}$
- 이때 목적함수는 $C(G)=-\log 4 + 2\,\mathrm{JSD}(p_{data}\|p_g)$
- JSD는 두 분포가 같을 때만 0이므로, 글로벌 최적은 $p_g=p_{data}$이고 그때 $D=\frac{1}{2}$, $C(G)=-\log 4$

실제로는 $p_g$를 직접 최적화하는 게 아니라 MLP 파라미터 $\theta_g$를 움직이니까 이론처럼 보장은 안 된다. 그래도 MLP가 잘 되니까 이 구조를 쓴다고 한다.

Figure 1 직관: $D$가 두 분포를 나누는 쪽으로 학습되고, $G$는 $D$가 진짜라고 볼 만한 영역으로 샘플을 옮긴다. 충분히 돌리면 두 분포가 겹친다.

---

## 실험 구성과 결과는? (Experiments)

MNIST, TFD, CIFAR-10. $G$는 ReLU+sigmoid, $D$는 maxout. dropout은 $D$에만 넣었다. $G$의 중간층 노이즈는 안 쓰고, 맨 아래 입력 $z$만 노이즈다.

likelihood를 직접 계산할 수 없어서, $G$ 샘플에 Gaussian Parzen window를 맞추고 테스트셋 log-likelihood를 근사한다. $\sigma$는 validation으로 고른다. 분산이 크고 고차원에서 약하지만, 샘플만 나오는 모델에서 쓸 수 있는 방법이라고 한다.

Parzen log-likelihood (Table 1):

- MNIST: DBN 138, Stacked CAE 121, Deep GSN 214, GAN **225**
- TFD: DBN 1909, Stacked CAE 2110, Deep GSN 1890, GAN 2057 (Stacked CAE가 더 높음)

샘플 그림(Figure 2)은 학습셋 nearest neighbor를 옆에 붙여서 단순 복사가 아님을 보여 준다. Markov chain이 없어서 샘플끼리 상관도 없다. Figure 3은 $z$ 공간을 직선으로 보간하면 숫자가 부드럽게 바뀌는 걸 보여 준다. 샘플이 기존 방법보다 확실히 낫다고 주장하진 않고, 경쟁할 만하다고만 한다.

---

## 무엇을 발견했고 한계점은 무엇인가? (Discussion)

장점: Markov chain이 필요 없다. 그래디언트는 backprop만 쓴다. 학습 때 inference가 필요 없다. 미분 가능한 함수면 $G, D$로 쓸 수 있다. 생성기가 데이터를 직접 복사하지 않고 $D$를 통과한 그래디언트만 받아서, 입력이 파라미터에 그대로 안 박힌다. MCMC 기반 모델은 mixing 때문에 분포가 좀 흐려야 하는데, GAN은 뾰족한 분포도 표현할 수 있다.

단점: $p_g(x)$를 식으로 안 쓴다. likelihood를 직접 못 구한다. $D$와 $G$ 속도를 맞춰야 한다. $G$만 너무 학습하면 여러 $z$가 같은 $x$로 가서 다양성이 죽는다. 논문은 이걸 Helvetica scenario라고 부른다. 나중에 mode collapse라고 부르는 현상의 원형이다.

평가도 약하다. Parzen window는 저자들도 분산이 크고 고차원에 약하다고 인정한다.

---

## 결론 및 주요 요약은? (Conclusion)

적대적 생성 모델이 실제로 돌아간다는 걸 보였다. 앞으로 확장할 수 있는 방향으로는 조건 생성 $p(x|c)$, $x$에서 $z$를 예측하는 inference 네트워크, 준지도 학습에 판별기 feature 쓰기, $G$와 $D$를 더 잘 맞추는 학습 방법 등을 적어 뒀다.

---

## 기존 연구 대비 차별점은 무엇인가?

RBM/DBM/DBN은 partition function이나 MCMC가 필요하다. GSN은 샘플링이 Markov chain이다. NCE는 노이즈 분포가 고정이다. VAE는 추론 네트워크로 ELBO를 쓴다.

GAN은 생성기와 판별기를 같이 학습하고, 샘플은 $z \to G(z)$ 한 번이면 된다. 학습/생성에 체인과 추론이 없다.

---

## 핵심 아이디어는?

생성 분포를 likelihood로 맞추지 않고, “가짜를 진짜와 구분 못 하게 만들기”로 맞춘다. 최적에서 $p_g=p_{data}$가 되고, 그때 목적함수는 두 분포의 Jensen–Shannon divergence가 된다.

---

## 주요 수식과 코드 분석

`code/gan.py`에서 value function이랑 최적 $D$, JSD 관계를 숫자로 확인했다.

$G$가 고정일 때 $D$의 목적

$$
\int p_{data}(x)\log D(x)+p_g(x)\log(1-D(x))\,dx
$$

는 $D^*(x)=\frac{a}{a+b}$에서 최대다. $a=p_{data}(x)$, $b=p_g(x)$.

이걸 $C(G)$에 넣으면

$$
C(G)=-\log 4 + 2\,\mathrm{JSD}(p_{data}\|p_g)
$$

이므로 $p_g=p_{data}$일 때만 최소고, 값은 $-\log 4 \approx -1.386$이다.

실습에서 $G$ 목적만 $\log(1-D)$로 두면 초반에 그래디언트가 죽는 것도 같이 봤다. $\log D(G(z))$로 바꾸면 가짜를 아직 잘 구분하는 $D$에서도 $G$가 움직일 여지가 있다.

실행:

```
python papers/code/gan.py
```
