# BiLM2_HAR.ipynb 코드 설명

> **VAE(Variational Autoencoder) 구조를 기반, Latent Space에서 신체 역학적(Biomechanical) 제약을 걸어 물리적으로 타당한 움직임을 학습하도록 유도하는 모델**
> 

*Q.잠재 공간(Latent Space)란?*

*A. 데이터의 본질을 담고있는 압축된 **지도**. 정보의 차원이 압축된 공간 → 이 공간은 단순히 정보의 압축만이 아닌 압축되기 전 데이터의 본질적인 의미들이 담긴다.* 

 *예를들어, 1000개가 넘는 숫자를 단 64개 숫자로 요약하면 이 공간에는 단순히 데이터의 압축 정도뿐만 아니라, “사람이 걷고 있다.”, “속도가 빠르다”, “다리가 굽혀졌다” 같은 함축적인 정보가 담기게 된다. GAP도 비슷한 관점에서 해석 가능하지만 이건 압축된 **강도**.*

## 0. VAE (Variational Autoencoder)

![image.png](image.png)

---

- 데이터를 단순히 압축 → 복원하는 것을 넘어서 데이터의 확률분포(Distribution)를 학습하는 모델.
- Generative AI의 시초가 되는 중요한 모델 (2013)

---

### (1) 일반 AE의 한계와 VAE 등장

- **일반 AE**
    - 입력 데이터를 압축 → 고정된 좌표 하나(z)로 보낸다.
    - 문제점: 잠재 공간이 Discontinuous!
    - Ex) 숫자 '1'이 좌표 `(10, 0)`에 있고, '7'이 `(20, 0)`에 있다고 가정. 그 중간인 `(15, 0)`을 디코더에 넣으면? '1'과 '7'이 섞인 숫자가 나오는 게 아니라, **의미 없는 노이즈**가 나온다. 학습하지 않은 빈 공간이기 때문.
- **VAE**
    - 입력 데이터를 좌표 점 하나가 아닌 확률분포로 만든다.
    - 즉, “이 데이터는 대략 여기($\mu$)쯤에 있는데, 이 정도 범위($\sigma$)내에 존재할 확률이 높아”라고 학습
    - 잠재 공간이 Continuous! → 중간값을 찍어도 의미있는 결과가 나온다.

### (2) VAE의 핵심 메커니즘 3단계

- Encoder($q_\phi$), Sampling, Decoder($p_\theta$)로 구성

**1단계: Encoder - 평균과 분산을 예측**

입력 x가 들어오면, encoder는 잠재 변수 z의 평균과 분산을 예측한다. ($x$ → $\mu,\ \sigma^2$)

**2단계: Reparameterization Trick**

분포에서 z를 랜덤하게 뽑아야 하는데, random sampling은 backprop이 불가능. (random에는 기울기가 없기 때문) 

$$
z=\mu + \sigma \ \odot \ \epsilon

$$

- $\epsilon$: 표준 정규분포에서 뽑은 노이즈 상수
- Ex) 기존 방식 = “이 분포에서 숫자 하나 뽑기” → 재파라미터화 = “무조건 0 근처에서 숫자하나 뽑고 그 수를 $\sigma$배만큼 늘리고, $\mu$만큼 이동시키기”
    
    ⇒ 이제 단순 사칙연산이 되어 미분이 가능해진다! 
    

**3단계: Decoder - 복원**

sampling된 z를 받아서 원래 데이터 x’으로 복원한다. 

### (3) Loss Function

- VAE의 학습 목표는 두 가지 상충되는 목표의 합 (ELBO: Evidence Lower Bound)

$$
Loss=Reconstruction \ Loss \ + \ KL\ Divergence
$$

1. **Reconstruction Loss (복원 오차)**
- “입력과 출력이 같아야한다.”
- latent space의 distribution이 데이터의 특징을 잘 표현하도록 돕는다.
1. **KL Divergence (쿨백-라이블러 발산)**
- “latent space의 distribution이 표준 정규분포(평균 0, 분산 1)와 비슷해야한다.”
- 분포가 너무 제멋대로 퍼지거나, 한 곳에만 뭉치지 않게 regularization 효과
- 이게 없으면 encoder는 분산을 0으로 만들어버리고 일반 AE와 동일해진다.
    
    *Q. 왜 없으면 분산을 0으로 만들까?*
    
    *A. only reconstruction만을 잘하려고 하기 때문. 분산=1이 곧 “노이즈 추가”의 형태로 작용하는것으로 복원 성능은 약간 희생하지만 (1) 새로운 데이터 생성 (2) 일반화 향상 이득이 있다. (VAE의 Trade-off 지점)* 
    
    ⇒ 결국 모델은 “적당한 노이즈($\sigma \approx 0.1 \sim 0.8)$을 유지하면서 그 노이즈 속에서도 최대한 복원을 잘하려고 하는 최적의 잠재 공간을 스스로 찾아낸다.”
    

---

# 1. Base settings

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
from scipy import signal

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
LATENT_DIM = 64        # 잠재 벡터(압축된 특징)의 크기
HIDDEN_DIM = 128       # 은닉층 노드 수
NUM_CLASSES = 6        # 분류할 행동의 개수 (걷기, 계단오르기 등)
SEQUENCE_LENGTH = 128  # 시계열 데이터의 길이 (Time steps)
INPUT_CHANNELS = 9     # 입력 데이터의 채널 수 (센서 축 개수)

# Biomechanical constraint parameters
JOINT_ANGLE_LIMITS = {'min': -np.pi, 'max': np.pi} # 관절 각도 제한 (-180도 ~ 180도)
VELOCITY_LIMIT = 10.0      # 속도 제한 임계값
ACCELERATION_LIMIT = 50.0  # 가속도 제한 임계값
```

- `JOINT_ANGLE_LIMITS`: 관절 각도를 [-π, π]로 제한하는 범위.
    - latent vector → 관절 각도로 매핑할 때 이 범위 안으로 scaling
    
    *Q. 관절각도를 π로 주는 이유는?*
    
    *A. (1) 각도의 주기는 2π로 보통 [0, 2π] 또는 [-π, π] 사용                     (2) `tanh` 출력이 [-1, 1] → scaling의 편의성                             (3) 실제 인간의 관절 범위와는 다름 (무릎: 130°, 팔꿈치: 150° 등) → “진짜         관절”이 아니라 가상의 관절 각도로 해석 → 각도의 변화가 너무 커지지않게          regularization을 주려는 용도로 쓰임.* 
    
    *⇒ 추후 각 관절마다 현실적인 범위 setting도 가능.* 
    
- `VELOCITY_LIMIT`: 관절 속도의 허용 상한
    - 실제로 계산된 속도(norm)가 이걸 넘으면 패널티
- `ACCELERATION_LIMIT`: 관절 가속도의 허용 상한
    - 실제로 계산된 가속도(norm)가 이걸 넘으면 패널티

# 2. Data Loading & time-series reconstruction

```python
class UCIHARDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.split = split

        if split == 'train':
            X_file = os.path.join(data_path, 'train', 'X_train.txt')
            y_file = os.path.join(data_path, 'train', 'y_train.txt')
        else:
            X_file = os.path.join(data_path, 'test', 'X_test.txt')
            y_file = os.path.join(data_path, 'test', 'y_test.txt')

        # Load data
        self.X = np.loadtxt(X_file)
        self.y = np.loadtxt(y_file) - 1  # Convert to 0-indexed

        # Reshape to (samples, channels, timesteps)
        # UCI-HAR has 561 features from 128 timesteps × 9 channels
        # We need to reconstruct the time series
        self.X = self._reconstruct_timeseries(self.X)

        self.X = torch.FloatTensor(self.X)
        self.y = torch.LongTensor(self.y.astype(np.int64))

        print(f"{split} set: {self.X.shape[0]} samples, X shape: {self.X.shape}")

    def _reconstruct_timeseries(self, X):
        """Reconstruct time series from UCI-HAR features"""
        # For simplicity, we'll load the raw Inertial Signals
        # This is a placeholder - actual implementation would load body_acc, body_gyro files
        batch_size = X.shape[0]

        # Create synthetic time series from features
        # In practice, load from Inertial Signals folder
        timeseries = np.zeros((batch_size, INPUT_CHANNELS, SEQUENCE_LENGTH))

        # Use FFT to create time series from frequency features
        for i in range(batch_size):
            for c in range(INPUT_CHANNELS):
                start_idx = c * 64
                end_idx = start_idx + 64
                if end_idx <= X.shape[1]:
                    freq_features = X[i, start_idx:end_idx]
                    # Pad and inverse FFT
                    padded = np.pad(freq_features, (0, SEQUENCE_LENGTH - len(freq_features)))
                    timeseries[i, c, :] = np.real(np.fft.ifft(padded))

        return timeseries

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

- 현재 코드에선 `X_train.txt`, `X_test.txt` 를 불러와서 미리 전처리된 합성 파일을 사용하고 있음. 이를 `_reconstruct_timeseries` 함수 내에서 각 channel당 64개 feature가져오고 이걸 주파수 성분이라고 가정하고 `ifft`로 시간축 신호처럼 만들고 있다. (테스트용도로 짜여진 코드)

→ 정석대로 `body_acc_x_train.txt`, `body_gyro_x_train.txt` 등을 사용하는 코드로 교체

```python
class UCIHARDataset(Dataset):
    def __init__(self, data_path, split='train'):
        self.split = split

        # 라벨 파일 경로
        y_file = os.path.join(data_path, split, f'y_{split}.txt')
        self.y = np.loadtxt(y_file).astype(int) - 1  # [0..5]로 맞추기

        # Inertial Signals 폴더
        signals_path = os.path.join(data_path, split, 'Inertial Signals')

        # 사용할 9개 파일 이름
        signal_files = [
            f'body_acc_x_{split}.txt',
            f'body_acc_y_{split}.txt',
            f'body_acc_z_{split}.txt',
            f'body_gyro_x_{split}.txt',
            f'body_gyro_y_{split}.txt',
            f'body_gyro_z_{split}.txt',
            f'total_acc_x_{split}.txt',
            f'total_acc_y_{split}.txt',
            f'total_acc_z_{split}.txt',
        ]

        signals = []
        for fname in signal_files:
            file_path = os.path.join(signals_path, fname)
            # 각 파일 shape: (N_samples, 128)
            sig = np.loadtxt(file_path)
            signals.append(sig)

        # signals: 리스트 길이 9, 각 원소가 (N, 128)
        # → (9, N, 128) → (N, 9, 128) 로 바꾸기
        X = np.stack(signals, axis=1)  # (N, 9, 128)

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(self.y)

        print(f"{split} set: {self.X.shape[0]} samples, X shape: {self.X.shape}")  # (N, 9, 128)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

# 3. **BiomechanicalConstraintModule**

- latent 공간에 “물리 제약” 걸기

```python
class BiomechanicalConstraintModule(nn.Module):
    """Applies biomechanical constraints to latent representations"""
    def __init__(self, latent_dim, num_joints=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_joints = num_joints
```

- latent vector ‘z’를 “가상의 관절 상태”라고 가정하고 ‘z’가 시간에 따라 너무 이상하게 움직이지 않도록 추가적인 손실을 계산하는 블록
- `num_joints=32` : ‘z’안에 32개 관절의 상태를 encode한다고 가정한다는 의미
    - z를 그냥 숫자 32개짜리 벡터로 두지 않고 해석을 하나 더 부여하는 것. 어떤 시점의 “몸의 상태”를 압축해놓은 것이라고 생각. → z를 각 함수로 매핑
    - 일반 z = “마음대로 학습된 아무 의미 없는 벡터
    - z + biomech loss = “관절 상태와 동역학적으로 일관되게 연결된 벡터

### 3-1. latent → joint angle / velocity / acceleration 예측

```python
        # Joint angle mapping network
        self.joint_mapper = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_joints),
            nn.Tanh()  # Bound to [-1, 1]
        )

        # Velocity predictor - output same dimension as joints
        self.velocity_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_joints)
        )

        # Acceleration predictor - output same dimension as joints
        self.accel_net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_joints)
        )
```

- `joint_mapper(z)`
    - z → `num_joints` 차원(32) → `tanh` → [-1, 1]
    - 이걸 관절 각도로 해석
- `velocity_net(z)`
    - z에서 바로 관절 속도를 예측
    - 실제 finite difference로 계산한 속도와 비교하여 MSE Loss
- `accel_net(z)`
    - z에서 바로 관절 가속도를 예측
    - 마찬가지로 finite difference로 계산한 속도와 비교하여 MSE Loss

### 3-2. 각도/속도/가속도 계산용 함수

```python
    def compute_joint_angles(self, z):
        """Map latent to joint angles with physical constraints"""
        angles = self.joint_mapper(z)  # [-1, 1]
        # Scale to physical range
        angle_range = JOINT_ANGLE_LIMITS['max'] - JOINT_ANGLE_LIMITS['min']
        angles = angles * (angle_range / 2) + (JOINT_ANGLE_LIMITS['max'] + JOINT_ANGLE_LIMITS['min']) / 2
        return angles

    def compute_velocity(self, z):
        """Predict velocity from latent"""
        return self.velocity_net(z)

    def compute_acceleration(self, z):
        """Predict acceleration from latent"""
        return self.accel_net(z)
```

- `compute_joint_angles(z)`
    - `joint_mapper(z)` → [-1, 1] 범위
    - 이걸 `[-π, π]` 범위로 스케일링하여 ‘물리적인 각도처럼’ 사용
- `comput_velocity(z)` / `compute_acceleration(z)` : 각각의 MLP 통과 결과

### 3-3. `biomechanical_loss`

```python
    def biomechanical_loss(self, z_sequence):
        """
        Compute biomechanical constraint violation loss
        z_sequence: (batch, seq_len, latent_dim)
        """
        batch_size, seq_len, _ = z_sequence.shape
```

- 입력: `z_seqeunce`
    - `(B, T, D)` → D는 latent_dim
    - 시간에 따라 변하는 latent z를 묶어놓은 것.
    - 이 코드에서는 진짜 시간 시퀀스가 아니라, `z` 하나에다가 작은 noise를 T개 더해 만든 “가짜 시퀀스” 사용 중. → `train_bilm` 수정 예정

**(1) joint angle → velocity / acceleration (finite difference)**

```python
        # Joint angle consistency
        angles = self.compute_joint_angles(z_sequence.reshape(-1, self.latent_dim))
        angles = angles.reshape(batch_size, seq_len, -1)
```

- 각 시간 t마다 `z_t` → joint angles 계산
- `angles` shape = `(B, T, num_joints)`

```python
        # Compute velocity from finite differences of angles (ground truth)
        velocity_actual = angles[:, 1:] - angles[:, :-1]
```

- 실제 속도(가정) = `angle(t+1) - angle(t)`
- shape: `(B, T-1, num_joints)`

```python
        # Predict velocity from latent
        velocity_pred = self.compute_velocity(z_sequence[:, :-1].reshape(-1, self.latent_dim))
        velocity_pred = velocity_pred.reshape(batch_size, seq_len-1, -1)
```

- `z_t`를 넣어서 velocity_net이 예측한 속도.
- `velocity_pred`도 `(B, T-1, num_joints)`

```python
        # Velocity consistency loss
        velocity_diff = torch.mean((velocity_pred - velocity_actual) ** 2)
```

- “z를 보고 예측한 속도” vs “각도 차분으로 계산한 실제 속도”
    
    → MSE로 일치시키는 loss.
    
    *Q. 이산 차분이란?*
    
    *A. 미분을 직접하지 못할 때 근사해서 사용하는 것.* 
    
    $$
    vk≈ \frac {x_{k+1} - x_k}{Δt}
    $$
    

**(2) velocity → acceleration**

```python
        # Compute acceleration from finite differences of velocity (ground truth)
        accel_actual = velocity_actual[:, 1:] - velocity_actual[:, :-1]
```

- 마찬가지로 이산 차분:
    - `accel_actual = v(t+1) - v(t)`
- shape: `(B, T-2, num_joints)`

```python
        # Predict acceleration from latent
        accel_pred = self.compute_acceleration(z_sequence[:, :-2].reshape(-1, self.latent_dim))
        accel_pred = accel_pred.reshape(batch_size, seq_len-2, -1)

        # Acceleration consistency loss
        accel_diff = torch.mean((accel_pred - accel_actual) ** 2)

```

- `z_t` 를 보고 aceel_net이 예측한 가속도 vs Velocity 차분으로 계산한 가속도
    
    → MSE로 일치시키는 loss.
    

**(3) 속도/가속도의 “너무 큰 값” 제한**

```python
        # Velocity magnitude constraint
        velocity_magnitude = torch.norm(velocity_actual, dim=-1)
        velocity_violation = torch.relu(velocity_magnitude - VELOCITY_LIMIT).mean()

        # Acceleration magnitude constraint
        accel_magnitude = torch.norm(accel_actual, dim=-1)
        accel_violation = torch.relu(accel_magnitude - ACCELERATION_LIMIT).mean()
```

- `velocity_magnitude` = 각 joint 속도 벡터의 L2 norm.
- 그것이 VELOCITY_LIMIT(10.0)를 넘으면:
    - `relu(v_norm - limit)` → 초과분만 패널티.
- 가속도도 마찬가지.

**(4) 각도 변화가 너무 들쭉날쭉하지 않도록 (smoothness)**

```python
        # Joint angle smoothness (penalize jerky movements)
        angle_smoothness = torch.mean(velocity_actual ** 2)
```

- 사실상 `velocity_actual` 제곱의 평균 → 속도 에너지 같은 느낌.
- 이것도 크면 클수록 패널티 →
    
    joint angle이 시간에 따라 너무 급격하게 바뀌지 않게 유도.
    

(5) 전체 biomechanical loss 합치기

```python
        total_loss = (velocity_diff + accel_diff +
                     0.1 * velocity_violation + 0.1 * accel_violation +
                     0.01 * angle_smoothness)

        return total_loss
```

- `velocity_diff` : 예측 속도 vs 실제 속도(FD) 일관성
- `accel_diff` : 예측 가속도 vs 실제 가속도 일관성
- `velocity_violation` : 속도 넘치면 패널티
- `accel_violation` : 가속도 넘치면 패널티
- `angle_smoothness` : 각도가 너무 요동치지 않게

람다값들: → 추후 튜닝 

- 속도/가속도 일치 (diff) 는 1.0
- limit 초과 패널티는 0.1
- smoothness는 0.01

# 4. Encoder - **시계열 → latent µ, logvar**

```python
class Encoder(nn.Module):
    """Encode time series to latent space"""
    def __init__(self, input_channels, hidden_dim, latent_dim, seq_length):
        super().__init__()

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)

        self.pool = nn.MaxPool1d(2)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)

        # Calculate size after pooling
        pooled_length = seq_length // 8  # 3 pooling layers

        self.fc_mu = nn.Linear(128 * pooled_length, latent_dim)
        self.fc_logvar = nn.Linear(128 * pooled_length, latent_dim)

        self.dropout = nn.Dropout(0.3)
```

- 입력: `x` shape = **(B, C=9, L=128)**
- Conv1d 3번 + MaxPool1d(2) 3번 → 시간 길이가 `/2/2/2 = 1/8`로 줄어듦
    - 처음: 128
    - 1번째 풀링 후: 64
    - 2번째: 32
    - 3번째: 16
- 채널 수는 9 → 32 → 64 → 128로 늘어남.
- 마지막 feature shape: `(B, 128, L_pooled)` with `L_pooled = 128//8 = 16`

```python
    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        x = x.flatten(1)
        x = self.dropout(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar
```

- Conv + BatchNorm + ReLU + Pool 3층 → 시계열 패턴을 점점 요약
- `x.flatten(1)` → `(B, 128 * pooled_length)` = `(B, 128*16 = 2048)`
- `fc_mu`, `fc_logvar`: 둘 다 `(B, latent_dim=64)` 생성
- 결과: **µ (평균)**, **logvar (분산의 로그)** → VAE용 파라미터.

# 5. Decoder - latent z → 다시 시계열 복원

```python
class Decoder(nn.Module):
    """Decode latent to time series"""
    def __init__(self, latent_dim, hidden_dim, output_channels, seq_length):
        super().__init__()

        pooled_length = seq_length // 8

        self.fc = nn.Linear(latent_dim, 128 * pooled_length)
        self.pooled_length = pooled_length

        self.deconv1 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose1d(32, output_channels, kernel_size=4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

```

- Encoder에서 줄였던 걸 **정반대로** 복원하는 구조
- ConvTranspose1d(stride=2, kernel=4, padding=1) → 길이를 2배씩 키움:
    - 16 → 32 → 64 → 128
    
    *Q. ConvTranspose1d란?*
    
    *A. 보통의 Conv1d는 stride=2면 길이가 절반으로 줄어든다. ConvTranspose1d는 반대로 길이를 2배 늘린다.*
    

```python
    def forward(self, z):
        x = self.fc(z)
        x = x.reshape(-1, 128, self.pooled_length)

        x = torch.relu(self.bn1(self.deconv1(x)))
        x = torch.relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)

        return x
```

- `z` (B, 64) → FC → (B, 128*16) → reshape → (B, 128, 16)
- ConvTranspose 3번:
    - (B, 128, 16)
    - → (B, 64, 32)
    - → (B, 32, 64)
    - → (B, output_channels=9, 128)
- 최종 출력 shape: `(B, 9, 128)`
    
    → 입력과 같은 모양으로 복원.
    
- 인코더: 128 → 64 → 32 → 16
- 디코더: 16 → 32 → 64 → 128

# 6. BILMClassifier

```python
class BILMClassifier(nn.Module):
    """BILM: Biomechanical Inverse-Latent Modeling for HAR"""
    def __init__(self, input_channels, hidden_dim, latent_dim, num_classes, seq_length, num_joints=32):
        super().__init__()

        self.encoder = Encoder(input_channels, hidden_dim, latent_dim, seq_length)
        self.decoder = Decoder(latent_dim, hidden_dim, input_channels, seq_length)
        self.biomech_module = BiomechanicalConstraintModule(latent_dim, num_joints=num_joints)

        # Classifier on latent space
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
```

- `encoder` : 시계열 → (µ, logvar)
- `decoder` : latent z → 시계열 복원
- `biomech_module` : z 시퀀스에 물리 제약 loss
- `classifier` : z → 클래스 logits

```python
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
```

- reparameterization trick

```python
    def forward(self, x):
        # Encode
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)

        # Classify
        logits = self.classifier(z)

        return logits, x_recon, mu, logvar, z
```

- `x` (B, 9, 128)
    
    → Encoder → µ, logvar
    
- µ, logvar → reparameterize → **z (B, latent_dim=64)**
- z → Decoder → `x_recon` (B, 9, 128)
- z → Classifier → logits (B, 6)

# 7. Train

```python
def train_bilm(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        logits, x_recon, mu, logvar, z = model(data)
```

- `data`: `(B, 9, 128)`
- `model(data)`에서 나온 것들:
    - `logits` : 분류 결과 (B, 6)
    - `x_recon` : 복원된 시계열 (B, 9, 128)
    - `mu`, `logvar` : VAE latent 분포 파라미터
    - `z` : 샘플링된 latent

(1) 분류 손실 (CE)

```python
        # Classification loss
        ce_loss = nn.functional.cross_entropy(logits, target)
```

(2) 재구성 손실 (Recon MSE)

```python
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(x_recon, data)
```

- 디코더가 `z → x_recon`으로 복원한 결과와
- 원본 입력 `data`를 비교하는 **MSE**.

(3) KL loss (VAE regularization)

```python
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)
```

(4) Biomechanical constraint loss

```python
        # Biomechanical constraint loss
        # Create temporal latent sequence by adding temporal variations
        batch_size = z.size(0)
        seq_len = 10
        temporal_noise = torch.randn(batch_size, seq_len, z.size(1), device=z.device) * 0.1
        z_sequence = z.unsqueeze(1) + temporal_noise  # (batch, seq_len, latent_dim)

        biomech_loss = model.biomech_module.biomechanical_loss(z_sequence)
```

- `z` 하나에서 **노이즈로 가짜 시퀀스** 만들고
- 그걸 biomech 모듈에 넣어서
    - joint angle/velocity/acceleration 일관성
    - 속도/가속도 한계
    - smoothness
- 등을 포함한 **추가 물리 제약 loss**를 계산.

(5) 최종 Loss

```python
        # Total loss
        loss = ce_loss + 0.1 * recon_loss + 0.01 * kl_loss + 0.05 * biomech_loss
```

- `1.0 * ce_loss` → **가장 중요 (분류 성능)**
- `0.1 * recon_loss` → 재구성은 중요하지만, CE보다는 서브
- `0.01 * kl_loss` → latent를 너무 세게 구속하면 표현력이 떨어지니까 살짝만
- `0.05 * biomech_loss` → biomech 제약은 “보조 정규화” 느낌

```python
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
     
		    avg_loss = total_loss / len(train_loader)
		    accuracy = 100. * correct / total
		    return avg_loss, accuracy
```

- 역전파 & 정확도 계산 → 평균 loss와 정확도를 epoch 단위로 반환.

# 8. Evaluate

```python
def evaluate_bilm(model, test_loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            logits, x_recon, mu, logvar, z = model(data)

            ce_loss = nn.functional.cross_entropy(logits, target)
            recon_loss = nn.functional.mse_loss(x_recon, data)

            loss = ce_loss + 0.1 * recon_loss
            total_loss += loss.item()
```

- **CE + 0.1 * Recon**만 사용.
Q. 왜 evaluate시에는 biomech & KL을 사용하지 않을까?
    
    A. “훈련 시 정규화 역할”일 뿐. 결국 “분류가 잘 되냐?”만 확인하는 용도