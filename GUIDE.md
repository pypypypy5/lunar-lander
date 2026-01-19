# 강화학습 실습: Lunar Lander 학습시키기

## 🎯 실습 목표

이 실습에서는 **강화학습(Reinforcement Learning)**의 대표적인 두 알고리즘을 직접 구현하여 달 착륙선(Lunar Lander) 게임을 학습시킵니다.

- **DQN (Deep Q-Network)**: 가치 기반 학습
- **Actor-Critic**: 정책 기반 + 가치 기반 하이브리드 학습

## 📁 프로젝트 구조

```
lunar-lander/
├── agent_class.py          # ⭐ 여러분이 코드를 작성할 파일
├── train_agent.py          # 학습 실행 스크립트
├── run_agent.py            # 학습된 에이전트 평가 스크립트
└── GUIDE.md                # 이 파일
```

## 🚀 시작하기

### 1. 환경 설정

필요한 패키지를 설치하세요:

```bash
pip install gymnasium torch h5py numpy
```

### 2. 프로젝트 이해하기

#### Lunar Lander 환경
- **목표**: 달 착륙선을 안전하게 착륙시키기
- **상태(State)**: 8차원 벡터
  - 위치 (x, y)
  - 속도 (vx, vy)
  - 각도, 각속도
  - 왼쪽/오른쪽 다리 착지 여부
- **행동(Action)**: 4가지
  - 0: 아무것도 안 함
  - 1: 왼쪽 엔진 점화
  - 2: 메인 엔진 점화 (위로)
  - 3: 오른쪽 엔진 점화
- **보상(Reward)**:
  - 착륙 성공: +100~200점
  - 추락: -100점
  - 연료 사용: 감점
  - 목표 지점에 가까울수록: 보너스

#### 강화학습 핵심 개념

**Bellman 방정식**:
```
Q(s, a) = r + γ * max Q(s', a')
```
- `s`: 현재 상태
- `a`: 현재 행동
- `r`: 받은 보상
- `γ`: 할인율 (discount factor, 0.99)
- `s'`: 다음 상태
- `a'`: 다음 행동

## ✍️ 실습 과제

`agent_class.py` 파일에서 `TODO` 주석이 있는 4개의 함수를 구현해야 합니다.

---

### 📝 Part 1: DQN (Deep Q-Network)

#### 과제 1-1: `dqn.act()` - Epsilon-Greedy 행동 선택

**위치**: `agent_class.py` 약 781번째 줄

**목표**: Epsilon-greedy 전략으로 행동을 선택하는 함수를 구현하세요.

**개념**:
- **탐험(Exploration)**: 확률 ε로 랜덤 행동 선택 → 새로운 전략 발견
- **활용(Exploitation)**: 확률 (1-ε)로 최선의 행동 선택 → 학습된 지식 활용

**구현 가이드**:
```python
def act(self, state, epsilon=0.):
    if self.in_training:
        epsilon = self.epsilon

    # TODO: 여기에 코드 작성
    # 1. torch.rand(1).item()으로 0~1 사이 난수 생성
    # 2. 난수 > epsilon이면:
    #    - policy_net으로 Q값 계산
    #    - 가장 높은 Q값의 행동 선택 (argmax)
    # 3. 난수 ≤ epsilon이면:
    #    - 랜덤 행동 선택 (torch.randint 사용)
```

**힌트**:
- `self.neural_networks['policy_net']`로 네트워크 접근
- `torch.no_grad()` 블록 안에서 추론
- `policy_net.eval()` 모드로 전환
- `torch.tensor(state)`로 입력 변환
- `.argmax(0).item()`으로 행동 인덱스 추출

**테스트 방법**:
```python
# agent_class.py에서 NotImplementedError를 제거하고 구현 후
import agent_class as agent
params = {'N_state': 8, 'N_actions': 4}
my_agent = agent.dqn(params)
# 랜덤 상태로 테스트
import numpy as np
state = np.random.randn(8)
action = my_agent.act(state, epsilon=0.1)
print(f"선택된 행동: {action}")  # 0~3 사이 정수가 나와야 함
```

---

#### 과제 1-2: `dqn.run_optimization_step()` - Bellman 방정식 구현

**위치**: `agent_class.py` 약 829번째 줄

**목표**: DQN의 핵심인 Bellman 방정식을 사용한 Q-함수 학습을 구현하세요.

**개념**:
- **Policy Network**: 현재 학습 중인 Q-함수
- **Target Network**: 안정적인 타겟 Q값을 위한 복사본
- **Experience Replay**: 과거 경험을 저장하고 랜덤 샘플링하여 학습

**DQN vs Double DQN**:
- **Standard DQN**: Target network로 행동 선택 + 평가
  ```python
  Q_next = target_net(s').max()
  ```
- **Double DQN**: Policy net으로 행동 선택, Target net으로 평가 (과대평가 방지)
  ```python
  a_best = policy_net(s').argmax()
  Q_next = target_net(s')[a_best]
  ```

**구현 가이드**:
```python
def run_optimization_step(self, epoch):
    # 메모리에서 배치 샘플링 (이미 구현됨)
    state_batch, action_batch, next_state_batch, reward_batch, done_batch = ...

    # TODO: 여기에 코드 작성

    # [Step 1] LHS 계산 (현재 Q값)
    # LHS = policy_net(state_batch).gather(dim=1, index=action_batch.unsqueeze(1))

    # [Step 2] RHS 계산 (타겟 Q값)
    # if self.doubleDQN:
    #     # Double DQN: policy net으로 행동 선택
    #     argmax_next_state = policy_net(next_state_batch).argmax(dim=1)
    #     # target net으로 Q값 평가
    #     Q_next_state = target_net(next_state_batch).gather(...)
    # else:
    #     # Standard DQN: target net으로 직접 최댓값
    #     Q_next_state = target_net(next_state_batch).max(1)[0].detach()
    #
    # RHS = Q_next_state * self.discount_factor * (1 - done_batch) + reward_batch
    # RHS = RHS.unsqueeze(1)  # shape 맞추기

    # [Step 3] 손실 계산 및 최적화
    # loss_ = loss(LHS, RHS)
    # optimizer.zero_grad()
    # loss_.backward()
    # optimizer.step()
```

**핵심 포인트**:
- `done_batch`: 에피소드가 끝났으면 다음 상태의 Q값은 0
- `.detach()`: Target network의 gradient 계산 방지
- `.gather()`: 특정 인덱스의 값만 선택

---

### 📝 Part 2: Actor-Critic

#### 과제 2-1: `actor_critic.act()` - 확률적 정책

**위치**: `agent_class.py` 약 983번째 줄

**목표**: Actor 네트워크가 출력한 확률 분포에서 행동을 샘플링하세요.

**개념**:
- **DQN과의 차이점**:
  - DQN: 결정론적 (항상 Q값이 최대인 행동 선택)
  - Actor-Critic: 확률적 (확률 분포에서 샘플링)
- **장점**: 더 부드러운 탐험, 연속적인 정책 개선

**구현 가이드**:
```python
def act(self, state):
    actor_net = self.neural_networks['policy_net']

    # TODO: 여기에 코드 작성
    # with torch.no_grad():
    #     actor_net.eval()
    #     # 1. actor_net에 state 입력 → affinities(선호도) 출력
    #     # 2. Softmax로 확률 분포 변환
    #     probs = self.Softmax(actor_net(torch.tensor(state)))
    #     # 3. Categorical 분포 생성
    #     m = Categorical(probs)
    #     # 4. 샘플링
    #     action = m.sample()
    #     actor_net.train()
    #     return action.item()
```

**예시**:
```
State: [x=0.5, y=1.2, vx=-0.3, ...]
Actor Net 출력: [2.1, -0.5, 1.3, 0.8]  (affinities)
Softmax 후:     [0.55, 0.04, 0.26, 0.15]  (확률)
→ 55% 확률로 행동 0, 26% 확률로 행동 2 선택
```

---

#### 과제 2-2: `actor_critic.run_optimization_step()` - Actor-Critic 학습

**위치**: `agent_class.py` 약 1020번째 줄

**목표**: Actor와 Critic 두 네트워크를 동시에 학습시키세요.

**개념**:
- **Critic (가치 함수)**: 상태의 좋고 나쁨을 평가 → V(s)
- **Actor (정책)**: 어떤 행동을 할지 결정 → π(a|s)
- **Advantage**: A = V(s') - V(s) → "이 행동이 평균보다 얼마나 좋은가?"

**구현 가이드**:
```python
def run_optimization_step(self, epoch):
    # 배치 샘플링 (이미 구현됨)
    state_batch, action_batch, next_state_batch, reward_batch, done_batch = ...

    # TODO: 여기에 코드 작성

    # ========== Part 1: Critic 학습 ==========
    # critic_net.train()
    #
    # LHS = critic_net(state_batch)  # V(s)
    # Q_next_state = critic_net(next_state_batch).detach().squeeze(1)
    # RHS = Q_next_state * self.discount_factor * (1 - done_batch) + reward_batch
    # RHS = RHS.unsqueeze(1)
    #
    # loss = loss_critic(LHS, RHS)
    # optimizer_critic.zero_grad()
    # loss.backward()
    # optimizer_critic.step()
    #
    # critic_net.eval()

    # ========== Part 2: Actor 학습 ==========
    # actor_net.train()
    #
    # advantage_batch = (RHS - LHS).detach()  # Advantage 계산
    #
    # loss = loss_actor(state_batch, action_batch, advantage_batch)
    # optimizer_actor.zero_grad()
    # loss.backward()
    # optimizer_actor.step()
    #
    # actor_net.eval()
```

**핵심 포인트**:
- Critic을 먼저 학습 → 정확한 Advantage 계산
- Advantage가 양수: 평균보다 좋은 행동 → 해당 행동 확률 증가
- Advantage가 음수: 평균보다 나쁜 행동 → 해당 행동 확률 감소

---

## 🏃 실행하기

### 1. DQN 학습

```bash
# Standard DQN
python train_agent.py --f my_dqn --dqn --verbose

# Double DQN (추천!)
python train_agent.py --f my_ddqn --ddqn --verbose
```

**출력 예시**:
```
| episode | return          | minimal return      | mean return        |
|         | (this episode)  | (last 20 episodes)  | (last 20 episodes) |
|-------------------------------------------------------------------------|
|       0 |     -245.123    |       -245.123      |      -245.123      |
|       1 |     -189.456    |       -245.123      |      -217.290      |
...
|     523 |      234.567    |        201.234      |       231.890      |  ← 성공!
```

### 2. Actor-Critic 학습

```bash
python train_agent.py --f my_ac --verbose
```

### 3. 학습된 에이전트 평가

```bash
# DQN 평가 (100 에피소드)
python run_agent.py --f my_dqn --dqn --N 100 --verbose

# Actor-Critic 평가
python run_agent.py --f my_ac --N 100 --verbose
```

**출력 예시**:
```
Run 1 of 100 completed with return 245.3. Mean return over all episodes so far = 245.3
Run 2 of 100 completed with return 189.7. Mean return over all episodes so far = 217.5
...
Run 100 of 100 completed with return 251.2. Mean return over all episodes so far = 227.8
```

---

## 🐛 문제 해결 (Troubleshooting)

### ❌ NotImplementedError 발생
```python
NotImplementedError: TODO: DQN act 함수를 구현하세요
```
→ `agent_class.py`에서 해당 함수의 `raise NotImplementedError(...)`를 지우고 코드를 작성하세요.

### ❌ 학습이 너무 오래 걸림
- DQN은 보통 300~800 에피소드 소요 (10~30분)
- Actor-Critic은 약 30% 더 오래 걸림
- `--verbose` 플래그를 사용하여 진행 상황 확인

### ❌ 학습이 수렴하지 않음
1. **Epsilon 문제** (DQN):
   - 너무 빨리 감소: `d_epsilon`을 줄이기 (예: 0.00005 → 0.00003)
   - 너무 느리게 감소: `d_epsilon`을 늘리기

2. **Learning Rate 문제**:
   ```python
   # train_agent.py에서 수정
   'optimizer_args': {'lr': 1e-3}  # 1e-4로 줄이거나 1e-2로 늘리기
   ```

3. **Batch Size 문제**:
   ```python
   'batch_size': 32  # 16 또는 64로 변경
   ```

### ❌ Shape 에러
```python
RuntimeError: The size of tensor a (32) must match the size of tensor b (1)
```
→ `.unsqueeze(1)` 또는 `.squeeze(1)`로 차원 조정 필요

---

## 📊 학습 결과 분석

### 생성되는 파일
- `my_dqn.tar`: 학습된 모델 가중치
- `my_dqn_training_data.h5`: 학습 통계
  - 에피소드별 return
  - 에피소드별 duration
  - 총 학습 스텝 수
- `my_dqn_execution_time.txt`: 학습 소요 시간

### 좋은 학습의 지표
✅ 평균 return이 200 이상
✅ 에피소드가 짧아짐 (빠른 착륙)
✅ 안정적인 return (분산이 작음)

---

## 🎓 심화 학습

### 1. 하이퍼파라미터 튜닝
`train_agent.py`의 `parameters` 딕셔너리를 수정하여 다양한 설정 실험:
- Discount factor (0.95 ~ 0.99)
- Learning rate (1e-4 ~ 1e-2)
- Network architecture (hidden layers)

### 2. 알고리즘 비교
- 같은 하이퍼파라미터로 DQN vs Actor-Critic 비교
- 어떤 알고리즘이 더 빠르게 학습하나?
- 최종 성능 차이는?

### 3. 시각화
Jupyter Notebook을 사용하여:
- 학습 곡선 그래프
- Return 분포 히스토그램
- 에피소드 길이 변화

---

## 📚 참고 자료

### 논문
1. **DQN**: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) (Mnih et al., 2013)
2. **Double DQN**: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (van Hasselt et al., 2015)
3. **Actor-Critic**: [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) (Sutton & Barto)

### 개념 정리
- **Experience Replay**: 과거 경험을 저장하고 재사용 → 학습 안정화
- **Target Network**: 느리게 업데이트되는 네트워크 → 타겟 값 안정화
- **Epsilon-Greedy**: Exploration vs Exploitation 균형
- **Advantage**: 행동의 상대적 가치 평가

---

## ✅ 체크리스트

실습을 완료했다면 체크해보세요:

- [ ] `dqn.act()` 함수 구현 완료
- [ ] `dqn.run_optimization_step()` 함수 구현 완료
- [ ] DQN으로 에이전트 학습 성공 (평균 return > 200)
- [ ] `actor_critic.act()` 함수 구현 완료
- [ ] `actor_critic.run_optimization_step()` 함수 구현 완료
- [ ] Actor-Critic으로 에이전트 학습 성공
- [ ] 두 알고리즘 비교 분석

---

## 💬 질문이 있나요?

구현 중 막히는 부분이 있다면:
1. TODO 주석의 힌트를 다시 읽어보세요
2. PyTorch 공식 문서 참고: https://pytorch.org/docs/stable/index.html
3. Gymnasium 환경 문서: https://gymnasium.farama.org/environments/box2d/lunar_lander/

**Happy Learning! 🚀**
