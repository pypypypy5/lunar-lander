# Lunar Lander RL 프로젝트 - 코드베이스 참조 문서

## 프로젝트 개요
강화학습(RL)을 사용하여 달 착륙선(Lunar Lander) 게임을 학습하는 교육용 프로젝트.
학생들이 **DQN**과 **Actor-Critic** 알고리즘의 핵심 함수를 직접 구현.

## 핵심 파일 구조

```
lunar-lander/
├── agent_class.py          # 메인: RL 알고리즘 클래스 (학생 구현 필요)
├── train_agent.py          # 에이전트 학습 스크립트
├── run_agent.py            # 학습된 에이전트 평가 스크립트
├── GUIDE.md                # 학생용 실습 가이드 (한국어)
└── CLAUDE.md               # 이 문서 (개발자 참조용)
```

## agent_class.py 구조

### 클래스 계층
```
agent_base (기본 클래스)
├── dqn (Deep Q-Network)
└── actor_critic (Actor-Critic)
```

### 학생이 구현해야 할 함수 (TODO 주석으로 표시됨)

#### 0. agent_base 클래스 - Reward Function (line ~404)
- **`agent_base.compute_reward(state, action, next_state, env_reward, terminated, truncated, info)`**
  - 강화학습의 핵심: 보상 함수 설계
  - 환경 보상을 그대로 사용하거나, 커스텀 보상 설계 가능
  - 입력:
    * state: 현재 상태 [x, y, vx, vy, angle, angular_vel, left_leg, right_leg]
    * action: 선택한 행동 (0~3)
    * next_state: 다음 상태
    * env_reward: 환경에서 제공하는 기본 보상
    * terminated, truncated: 에피소드 종료 여부
    * info: 환경의 추가 정보
  - 출력: float (최종 보상)
  - 설계 고려사항:
    * 목표 지점과의 거리
    * 속도 제어 (너무 빠르면 추락)
    * 각도 제어 (수평 착륙)
    * 연료 소모 최소화
    * 양쪽 다리 착지

#### 1. DQN 클래스 (lines 686-912)
- **`dqn.act(state, epsilon)`** (line 781~819)
  - Epsilon-greedy 행동 선택
  - epsilon 확률로 랜덤, 1-epsilon 확률로 Q값 최대화 행동 선택

- **`dqn.run_optimization_step(epoch)`** (line 829~889)
  - Bellman 방정식 구현: `Q(s,a) = r + γ * max_a' Q(s',a')`
  - LHS: policy_net으로 현재 Q(s,a) 계산
  - RHS: target_net으로 타겟 Q값 계산
  - Double DQN: policy_net으로 행동 선택, target_net으로 평가
  - MSE Loss로 최적화

#### 2. Actor-Critic 클래스 (lines 916-1097)
- **`actor_critic.act(state)`** (line 983~1018)
  - 확률적 정책 (stochastic policy) 구현
  - Actor 네트워크가 출력한 확률 분포에서 행동 샘플링
  - Categorical 분포 사용

- **`actor_critic.run_optimization_step(epoch)`** (line 1020~1095)
  - Critic 학습: 가치 함수 V(s) 학습
    - Bellman 방정식으로 타겟 계산
  - Actor 학습: 정책 π(a|s) 학습
    - Advantage = V(s') - V(s) 계산
    - Policy gradient로 최적화

### 이미 구현된 인프라 (학생이 수정하면 안 됨)

#### agent_base 클래스 (lines 73-682)
- `train()`: 메인 학습 루프
- `get_samples_from_memory()`: Experience replay
- `evaluate_stopping_criterion()`: 학습 종료 조건
- HDF5 저장/로드 기능

#### 보조 클래스
- `memory` (lines 20-33): Experience replay buffer (deque 기반)
- `neural_network` (lines 35-69): 피드포워드 신경망

## 주요 파라미터

### 공통 파라미터
```python
'discount_factor': 0.99        # γ (감마)
'n_memory': 20000              # Replay buffer 크기
'training_stride': 5           # 몇 스텝마다 학습할지
'batch_size': 32               # 미니배치 크기
'n_episodes_max': 10000        # 최대 에피소드 수
```

### DQN 전용 파라미터
```python
'epsilon': 1.0                 # 초기 탐험 확률
'epsilon_1': 0.1               # 최종 탐험 확률
'd_epsilon': 0.00005           # Epsilon 감소율
'target_net_update_stride': 1  # Target net 업데이트 주기
'target_net_update_tau': 0.01  # Soft update 계수 (τ)
'doubledqn': False/True        # Double DQN 사용 여부
```

### Actor-Critic 전용 파라미터
```python
'affinities_regularization': 0.01  # L2 정규화 계수
```

## 환경 정보

### Lunar Lander (Gymnasium)
- **상태 공간**: 8차원 연속 (위치, 속도, 각도, 각속도, 다리 접촉 여부)
- **행동 공간**: 4개 이산 행동
  - 0: 아무것도 안 함
  - 1: 왼쪽 엔진 점화
  - 2: 메인 엔진 점화
  - 3: 오른쪽 엔진 점화
- **보상**: 착륙 성공 시 양수, 추락 시 음수, 연료 소모 시 감점

### 학습 종료 조건
- 최근 20 에피소드 중:
  - 최소 리턴 > 200
  - 평균 리턴 > 230

## 실행 방법

### DQN 학습
```bash
python train_agent.py --f my_dqn_agent --dqn --verbose
```

### Actor-Critic 학습
```bash
python train_agent.py --f my_ac_agent --verbose
```

### 에이전트 평가
```bash
python run_agent.py --f my_dqn_agent --dqn --N 100 --verbose
```

## 출력 파일

- `{name}.tar`: 학습된 모델 체크포인트
- `{name}_training_data.h5`: 학습 통계 (리턴, 에피소드 길이 등)
- `{name}_execution_time.txt`: 학습 소요 시간
- `{name}_trajs.tar`: 평가 결과

## 디버깅 팁

### NotImplementedError 발생 시
- 학생이 TODO 함수를 구현하지 않은 경우
- `agent_class.py`의 TODO 주석 확인

### 학습이 수렴하지 않을 때
- Epsilon decay가 너무 빠르거나 느릴 수 있음
- Learning rate 조정 필요 (`optimizer_args`)
- Batch size나 training_stride 조정

### 메모리 부족
- `n_memory` 줄이기
- `batch_size` 줄이기
- CPU 모드 유지 (line 11: `device = torch.device("cpu")`)

## 코드 수정 시 주의사항

1. **절대 수정하면 안 되는 부분**:
   - `agent_base` 클래스의 `train()` 메서드
   - `memory` 클래스
   - `neural_network` 클래스
   - HDF5 저장/로드 로직

2. **학생이 수정해야 하는 부분**:
   - `agent_base.compute_reward()` ⭐ 보상 함수 설계
   - `dqn.act()`
   - `dqn.run_optimization_step()`
   - `actor_critic.act()`
   - `actor_critic.run_optimization_step()`

3. **선택적으로 조정 가능한 부분**:
   - 하이퍼파라미터 (train_agent.py의 `parameters` 딕셔너리)
   - 신경망 구조 (`layers` 파라미터)
   - 학습 종료 조건 임계값

## 알고리즘 비교 (원본 프로젝트 결과)

- **DQN**: 평균 28% 빠른 학습, 약간 높은 평균 성능
- **Actor-Critic**: 더 넓은 분산, 최고 성능의 에이전트는 Actor-Critic
