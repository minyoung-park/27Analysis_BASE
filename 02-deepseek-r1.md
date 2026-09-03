# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

DeepSeek-AI, arXiv:2501.12948v2

베이스 모델은 DeepSeek-V3-Base (671B MoE, 토큰당 37B).

---

## 어떤 문제를 해결하려 했는가? (Abstract)

LLM이랑 CoT 프롬프트 덕분에 기초 추론은 좋아졌지만, 사람 주석 시연에 많이 의존하고 어려운 문제는 아직 약하다.

이 논문은 사람 추론 궤적 없이, 순수 강화학습만으로 추론 능력을 끌어낼 수 있다고 한다. 학습하다 보면 자기반성, 검증, 전략 전환 같은 행동이 생긴다. 수학, 코딩 대회, STEM처럼 답이 검증 가능한 과제에서 사람 시연 SFT보다 잘 나온다. 큰 모델에서 나온 추론 패턴은 작은 모델로 증류할 수 있다.

---

## 연구 동기와 문제점은 무엇인가? (Introduction)

추론은 수학, 논리, 프로그래밍의 핵심이다. 스케일을 키우면 능력이 나온다는 이야기와, “Let’s think step by step”처럼 CoT를 쓰면 중간 단계를 적어서 성능이 오른다는 이야기가 있다. 후학습에 고품질 다단계 궤적을 넣는 것도 도움이 된다.

문제는 두 가지다. 사람 주석 CoT는 비용이 크고 편향이 들어간다. 그리고 사람 사고방식을 복사하면 상한이 사람이 된다.

그래서 SFT 없이 RL로 자기진화를 시켜 본다. V3-Base 위에 GRPO를 쓰고, 보상은 최종 답이 맞는지로만 준다. 추론 과정을 어떻게 쓰라고는 안 정한다. 사람 패턴이 탐색을 막을 수 있다는 가설 때문에 RL 앞에 SFT를 안 넣었다.

이렇게 나온 게 DeepSeek-R1-Zero다. 응답이 길어지고 검증이랑 반성이 생긴다. 다만 가독성이 나쁘고, 영어/중국어가 한 CoT 안에서 섞이고, 글쓰기나 오픈도메인 QA는 약하다.

그래서 DeepSeek-R1은 rejection sampling, RL, SFT를 여러 단계로 섞어서 Zero의 추론은 살리면서 사람 선호에 맞춘다. 그다음 작은 모델로 증류해서 공개한다.

---

## 관련 연구 동향은 어떠한가? (Related Works)

본문에 Related Work 절은 없고 부록에 있다.

보통 후학습은 SFT 다음에 RLHF다. SFT는 빠르지만 사람 답에 묶인다. 이 논문은 SFT가 추론 탐색을 방해할 수 있다고 본다. 사람 답은 반성이나 검증을 빼먹는 경우가 많다.

CoT, self-consistency, Tree-of-Thoughts 같은 방법은 프롬프트로 추론을 구조화한다. inference-time scaling은 여러 경로를 뽑거나 MCTS, self-correct를 쓴다. R1은 학습 때 RL을 돌리고, 추론 때는 생각 토큰을 늘리는 쪽으로 스케일한다.

추론용 RL은 STaR처럼 맞은 CoT만 다시 SFT하거나, PRM으로 과정 보상을 주는 식이 많았다. R1-Zero는 outcome 보상만 base 모델에 바로 준다.

초기에 실패한 것도 적어 뒀다. PRM은 스텝을 나누기 어렵고 중간 정오 판정이 어렵고 reward hacking이 난다. MCTS는 토큰 공간이 너무 넓어서 value model이 약하면 막힌다.

---

## 연구 접근법과 모델 구조는 어떻게 되는가? (Method)

새 Transformer를 제안하는 논문은 아니다. 이미 있는 V3-Base 위에서 학습 신호를 바꾼다. V3-Base는 수학이랑 코드가 많은 사전학습을 해서, 후보 답을 어느 정도 샘플할 수 있다. RL은 그중 좋은 걸 고르는 쪽에 가깝다.

**GRPO**

PPO의 value model(critic)을 뺀 알고리즘이다. critic은 policy만큼 크고, 긴 CoT에서는 중간 토큰의 value를 예측하기가 어렵다. 앞에서 틀린 말을 뒤에서 고치기 때문이다.

질문 \(q\)마다 old policy에서 \(G\)개 답을 뽑고 아래를 최대화한다.

\[
J_{GRPO}(\theta)=\mathbb{E}\Big[\frac{1}{G}\sum_i \min\big(r_i(\theta)A_i,\ \mathrm{clip}(r_i(\theta),1-\varepsilon,1+\varepsilon)A_i\big)-\beta D_{KL}(\pi_\theta\|\pi_{ref})\Big]
\]

\[
r_i(\theta)=\frac{\pi_\theta(o_i\mid q)}{\pi_{\theta_{old}}(o_i\mid q)}
\]

Advantage는 그룹 안에서 표준화한다.

\[
A_i=\frac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)}
\]

PPO는 토큰마다 KL을 빼서 응답 길이를 간접적으로 처벌할 수 있다. 긴 CoT를 키우려면 불리하다. GRPO는 KL을 loss에 넣고, 400 step마다 reference를 최신 정책으로 바꾼다.

R1-Zero 설정: LR 3e-6, KL 0.001, 질문당 16개 샘플, max length는 8.2k step 이후 65536. 총 10400 step.

템플릿은 `<think>...</think>`, `<answer>...</answer>`만 정한다. 안에 무엇을 쓸지는 안 정한다.

**보상**

\[
R_{rule}=R_{acc}+R_{format}
\]

수학은 최종 답을 규칙으로 채점하고, 코드는 테스트케이스로 채점한다. 포맷 보상은 think 태그를 썼는지다. 뉴럴 RM은 추론에 안 쓴다. reward hacking이 잘 나서다.

이 상태로 학습하면 AIME 2024 pass@1이 15.6%에서 77.9%로 오른다. 응답 길이도 같이 길어진다. wait, mistake, verify 같은 단어가 늘고, 풀다가 다시 생각하는 aha moment가 나온다. MATH 레벨 5는 0.55에서 0.90까지 오른다. 쉬운 문제는 일찍 배우고 어려운 문제에서 RL 이득이 크다.

**DeepSeek-R1 파이프라인**

Zero는 읽기 힘들고 언어가 섞인다. R1은 단계를 나눈다.

1. cold-start SFT: 사람이 읽기 좋은 thinking 데이터 수천 개
2. 1차 RL: 규칙 보상 + 언어 일치 보상. 언어 보상은 추론 점수를 조금 깎지만 가독성이 좋아진다.
3. rejection sampling + SFT: 추론이랑 비추론(글쓰기, QA)을 같이 학습
4. 2차 RL: 추론은 규칙, 일반 대화는 helpful/safety RM. helpful은 최종 요약만 보고, safety는 사고 과정까지 본다. RM은 마지막 400 step만 쓴다. 더 돌리면 hacking이 난다.

**증류**

R1이 만든 80만 궤적으로 Qwen, Llama를 SFT한다. 작은 모델에 같은 순수 RL을 돌리는 것보다 증류가 낫다. 다만 성능 상한을 올리려면 큰 base에서 RL을 하는 쪽이 필요하다고 한다.

---

## 실험 구성과 결과는? (Experiments)

MMLU, GPQA, AIME 2024, MATH-500, LiveCodeBench, Codeforces, SWE-Bench, IF-Eval, AlpacaEval 등. long CoT에 greedy를 쓰면 반복이 심해서 temperature 0.6으로 여러 번 샘플한 뒤 pass@1을 평균낸다. few-shot은 R1 성능을 떨어뜨려서 zero-shot을 권장한다.

단계별로 보면 cold-start(Dev1)에서 지시 따르기는 오르고 AIME는 77.9 → 59.0으로 떨어진다. SFT가 추론을 상하게 할 수 있다는 가설이랑 맞는 장면이다. 그다음 추론 RL을 다시 돌리면 수학/코드가 회복되고, 마지막에 일반 데이터 RL을 넣으면 AlpacaEval이 62.1 → 87.6, ArenaHard가 75.6 → 92.3으로 오른다.

최종 R1: AIME 79.8, MATH-500 97.3, LiveCodeBench 65.9, Codeforces percentile 96.3. o1-1217이 AIME 79.2, MATH-500 96.4라서 수학/알고리즘은 비슷한 수준이다. SWE나 Aider는 o1이 더 낫다. 소프트웨어 엔지니어링은 평가가 느려서 RL을 많이 못 돌렸다고 한다.

GPT-4o에 majority vote를 많이 해도 갭이 안 줄어든다. AIME에서 4o는 64샘플로 9.3 → 13.4다. 샘플이 서로 독립이라 한 경로 안에서 고치지 못해서다.

증류: Qwen-1.5B distill이 AIME 28.9로 GPT-4o(9.3)보다 높다. 32B distill은 72.6이고, 같은 크기 모델에 RL만 돌린 Qwen2.5-32B-Zero는 47.0이다.

---

## 무엇을 발견했고 한계점은 무엇인가? (Discussion)

Base가 작으면 안 된다. 7B, 16B MoE는 AIME가 안 올랐고 길어지면 반복만 했다. 32B 이상에서 순수 RL이 먹혔다.

보상이 믿을 수 있어야 한다. 수학/코드는 규칙으로 막을 수 있는데, 글쓰기처럼 주관적인 과제는 해킹이 난다.

SFT와 RL 역할이 다르다. RL이 없으면 긴 반성이 안 생기고, SFT가 없으면 보상 정의가 애매한 과제에서 무너진다.

한계로 적어 둔 것: 도구(검색, 계산기)를 못 씀, 쉬운 문제에서 overthinking, 중/영 말고 다른 언어에서 언어 혼용, few-shot에 약함, SWE 성능이 V3 대비 큰 이득이 없음. 규칙이 없는 과제에 순수 RL을 키우는 건 아직 미해결이다.

안전은 GPT-4o 정도고, 추론이 강해지면 위험한 계획의 실행 가능성도 올라간다고 적어 뒀다.

---

## 결론 및 주요 요약은? (Conclusion)

사전학습 체크포인트 안에 복잡한 추론 잠재력이 이미 있고, 그걸 여는 건 사람 주석이 아니라 어려운 문제, 믿을 수 있는 verifier, 충분한 RL 연산이라고 정리한다. 답이 검증 가능한 과제는 이 방식으로 사람을 넘을 수 있고, 보상을 정의하기 어려운 과제가 남은 문제다.

---

## 기존 연구 대비 차별점은 무엇인가?

기존에는 사람 CoT를 SFT로 먼저 넣고 RL을 돌리는 경우가 많다. R1-Zero는 SFT 없이 outcome RL만 한다. 과정 감독(PRM)도 안 쓴다. 알고리즘은 critic 없는 GRPO다. 작은 모델은 직접 RL보다 큰 모델 궤적을 증류한다. 추론 보상은 규칙이고, 일반 대화 RM은 짧게만 쓴다.

---

## 핵심 아이디어는?

어떻게 생각하라고 가르치지 않고, 맞았는지만 알려 준다. 포맷 태그랑 최종 답 보상만 있어도 긴 CoT랑 반성이 생긴다. 그룹 안에서 상대 점수로 advantage를 계산하면 critic 없이도 긴 생성 RL이 된다. SFT는 가독성에는 이롭고 탐색에는 해로울 수 있다.

---
