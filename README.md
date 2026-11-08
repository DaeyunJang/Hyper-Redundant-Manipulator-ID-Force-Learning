# HRM 외력·접촉 위치 추정

CSV 입력으로 Fx/Fy/Fz와 접촉 ID를 학습하고, test 예측을 Excel과 그림으로 저장합니다.
실행 코드는 루트 Python 파일, 설정은 `configs/`, 데이터는 `datasets/train`과 `datasets/test`에 있습니다.
아래 명령은 프로젝트 루트의 같은 Bash 터미널에서 실행합니다.

## 내려받기와 환경 설치

Git, Git LFS, `uv`가 필요합니다. CSV 전체 다운로드 크기는 약 **3.63 GB**입니다.

```bash
git lfs install
git clone https://github.com/DaeyunJang/Hyper-Redundant-Manipulator-External-Force-Learning.git
cd Hyper-Redundant-Manipulator-External-Force-Learning
git lfs pull

bash scripts/setup_environment.sh
```

설치 스크립트는 Python 3.10 환경 `env_hrm_force_estimation`과 PyTorch 2.7.1/CUDA 12.6,
`requirements.txt`의 패키지를 설치합니다. GPU를 사용하려면 호환되는 NVIDIA 드라이버가 필요합니다.
이후 Python 명령은 `bash scripts/hrm_python.sh ...`로 실행합니다.

| 데이터 경로 | CSV 수 | 크기 |
|---|---:|---:|
| `datasets/train/train1.csv` ~ `train58.csv` | 58 | 약 3.57 GB |
| `datasets/test/test1.csv` | 1 | 약 66.65 MB |

파일 번호는 접촉 ID가 아닙니다. 라벨은 CSV의 `contact_segment_id`에서 읽습니다.
원래 파일명·접촉 ID·SHA256은 [datasets/file_names.csv](datasets/file_names.csv)에 기록했습니다.
현재 test 파일은 `test1.csv`입니다.
**ID18 데이터 한 개이므로 전체 body 접촉 위치를 평가하는 테스트 세트가 아닙니다.**
CSV 대신 몇 줄짜리 LFS 포인터만 보이면 `git lfs pull`을 다시 실행하세요.

## 설정과 데이터 확인

기본 설정은 [configs/train.yaml](configs/train.yaml)입니다. 실행 전에 다음 항목을 확인하세요.

- `input_columns`, `force_columns`, `id_column`: 사용할 CSV 열과 순서.
- `source_force_unit`, `force_scope`, `ft_sensor_calibration`: 힘 단위·학습 범위·무부하 판정 범위.
- `force_candidates`, `location_candidates`, `no_load_modes`: 학습할 모델과 방식.
- `window_samples`, `validation_fraction`, `batch_size`, `max_epochs`: 시간창·분할·학습 크기.

현재 기본 입력은 케이블 길이 4개, 장력 4개, 상대각 18개입니다.
기본 힘 범위는 `all_single_contact`, 시간창은 30행, validation은 각 train CSV의 뒤 20%입니다.
원본 CSV는 수정하지 않으며 선택한 입력·정답 열의 비수치/결측 행을 제외합니다.

```bash
# 모델 학습 없이 데이터와 분할만 확인합니다. 확인용 출력 폴더는 생성됩니다.
bash scripts/hrm_python.sh train.py --config configs/train.yaml --check-data

# YAML에 지정한 무부하 범위를 확인합니다.
bash scripts/hrm_python.sh analyze_no_load.py --config configs/train.yaml
```

입력 구성을 바꾸려면 `--config`에 다음 파일 중 하나를 지정합니다.

| 설정 파일 (`configs/` 아래) | 입력 |
|---|---|
| `train_cable_loadcell.yaml` | 케이블 길이 + 장력 |
| `train_cable_angle.yaml` | 케이블 길이 + 상대각 |
| `train_loadcell_angle.yaml` | 장력 + 상대각 |
| `train_cable_loadcell_angle.yaml` | 케이블 길이 + 장력 + 상대각 |
| `train_cable_loadcell_estimated_pose.yaml` | 케이블 길이 + 장력 + Estimated 자세 |
| `train_cable_loadcell_fk_tip_tf.yaml` | 케이블 길이 + 장력 + FK 자세 |

## 학습

처음에는 MLP 힘 모델과 MLP 위치 모델, `class0` 한 조합을 1 epoch 실행해 흐름을 확인하세요.
이 명령도 전체 지정 데이터의 준비와 학습·validation Excel 생성을 수행합니다.

```bash
HRM_RUN="run_$(date +%Y%m%d_%H%M%S)"
bash scripts/hrm_python.sh train.py --config configs/train.yaml \
  --models mlp --no-load-modes class0 --epochs 1 \
  --output "results/train/$HRM_RUN"
```

전체 비교가 필요할 때는 아래 명령을 별도로 실행합니다.
기본값은 **힘 11종 × 위치 11종 × 2방식 = 242조합**, 모델당 최대 60 epoch이므로
단일 조합 확인보다 시간과 저장 공간이 많이 필요합니다.

```bash
bash scripts/hrm_python.sh train.py --config configs/train.yaml \
  --output "results/train/${HRM_RUN}_all"
```

학습 폴더에는 조합별 `hrm_bundle.pt`, `validation.xlsx`, 비교표와 사용 설정이 저장됩니다.
다음 예측 명령은 첫 단일 조합 학습을 사용합니다. 전체 비교 모델을 사용하려면
`HRM_RUN="${HRM_RUN}_all"`로 바꾸세요. 새 실행에는 새로운 이름을 사용합니다.

## Test 예측과 Excel 정리

```bash
bash scripts/hrm_python.sh predict.py --train-dir "results/train/$HRM_RUN" \
  --output "results/predict/$HRM_RUN"

bash scripts/hrm_python.sh organize_results.py \
  "results/predict/$HRM_RUN/predictions.xlsx" --id-tolerance 2

bash scripts/hrm_python.sh analysis.py "results/predict/$HRM_RUN" \
  --output "results/analysis/$HRM_RUN"
```

`predict.py --train-dir`는 학습 폴더의 완료된 조합과 저장된 설정을 사용합니다.
저장된 `test_directory`에 현재 존재하는 CSV 전체를 읽으며, 현재 YAML로 모델 설정을 바꾸지 않습니다.
출력 폴더에는 `predictions.xlsx`, `predictions.csv`, `metrics.json` 등이 생성됩니다.

`organize_results.py`는 예측 Excel을 읽어 입력 옆 새 `organized_*` 폴더에 힘·ID Excel을 만듭니다.
`--id-tolerance 0`은 정확 ID, `2`는 ±2 기준입니다. 저장된 예측값은 바꾸지 않습니다.
`analysis.py`는 predict 폴더를 읽어 `force_and_id_all_combinations.xlsx`와 비교 그림을 만듭니다.
이 두 단계는 모델을 다시 실행하지 않습니다. 여러 입력 실행을 비교할 때는 폴더들을 함께 지정하세요.

```bash
bash scripts/hrm_python.sh analysis.py results/predict/실행A results/predict/실행B
```

예측과 Excel 정리를 한 번에 하려면 개별 명령 대신 아래 스크립트를 사용할 수 있습니다.

```bash
bash scripts/predict_and_organize.sh "results/train/$HRM_RUN" \
  "results/predict/${HRM_RUN}_organized" --id-tolerance 2
```

## MATLAB 그림과 주요 파일

MATLAB R2020b 이상에서 프로젝트 폴더를 열고 필요한 스크립트를 실행합니다.

```matlab
run('analyze_matlab.m')  % organize가 만든 force-estimation_*.xlsx 선택
run('plot_analysis.m')   % analysis가 만든 force_and_id_all_combinations.xlsx 선택
```

스크립트 상단에서 파일·ID 범위·그림 옵션을 바꿀 수 있습니다.
MATLAB은 저장된 Excel을 사용하며, 출력은 입력 Excel 옆 새 파일/폴더에 저장합니다.

| 파일 | 역할 |
|---|---|
| `train.py`, `model_zoo.py`, `data_utils.py` | 학습, 모델 정의, CSV 처리 |
| `predict.py` | 저장 모델의 test 예측 |
| `analyze_no_load.py` | 무부하 범위 확인과 선택 CSV 통계 |
| `organize_results.py`, `analysis.py` | 저장 Excel 정리와 분석 |
| `analyze_matlab.m`, `plot_analysis.m` | Excel 기반 MATLAB 분석·그림 |
| `configs/*.yaml`, `requirements.txt`, `scripts/*.sh` | 실행 설정, 의존성, 환경/실행 도구 |

모든 Python 도구는 `--help`로 실행 옵션을 확인할 수 있습니다.
