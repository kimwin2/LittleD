# 연구 인계서: LittleBit 단일 Draft로 Prefill·Decode를 함께 다루는 Speculative Co-design

> 이 문서는 다른 에이전트에게 연구를 넘기기 위한 정리본이다.
> 작성일: 2026-11-10 · 레포: `kimwin2/LittleD` (SpecForge 포크) · 작업 브랜치: `ccr-db33831f-56bgtr`

---

## 0. 한 줄 요약

**작은 LittleBit draft 하나가 (1) prefill 때는 프롬프트에서 중요한 토큰을 고르는 scorer 역할을 하고, (2) decode 때는 DFlash 방식으로 토큰을 미리 생성하는 draft 역할을 한다.**
draft는 프롬프트 전체 문맥을 들고 있고, 비싼 target은 draft가 골라준 일부 토큰만 본다(**역할 반전**). target이 버린 토큰이 decode 중에 필요해지면 draft가 감지해서 다시 넣는다(**recall**).

---

## 1. 배경

### 1.1 이전 연구 (이미 완료)
- DFlash draft(block diffusion 방식의 draft)를 **LittleBit**(low-rank + 이진 분해 기반의 1bit 미만 양자화)으로 양자화했다.
- 핵심 결과: **LLM 본체는 극한 양자화에서 무너지지만, draft는 잘 버틴다.**

### 1.2 우리가 가진 무기
- **LittleBit**: 0.1~1 bpw 수준의 극한 양자화. low-rank 구조라서 메모리뿐 아니라 **FLOPs도 줄어든다**. prefill은 연산이 병목이라 이 점이 특히 중요하다.
- **NPU / on-device**: on-device LittleBit linear가 이미 구현돼 있다(커밋 `ca6bcee`).

### 1.3 주제 선정 과정 (결정 완료, 다시 논의하지 말 것)
- MoE speculative decoding은 **기각**했다. 양자화 draft + expert prefetch + Flash-NPU 조합을 이미 MoE-SpeQ(2511.14102), DraftExpert(2607.24434), Speculating Experts(2603.19289)가 했다.
- **Speculative prefill 계열을 채택**했다. 이유:
  - scoring은 top-1을 맞힐 필요 없이 **top-k 집합만 겹치면 되는 "순위" 작업**이라, drafting보다 양자화에 더 강할 것이라는 가설이 지난 논문과 자연스럽게 이어진다.
- on-device에서 "NPU prefill은 이미 빠르다"는 반론이 있다(sd.npu, llm.npu). 그래서 **TTFT보다 KV 메모리와 decode 대역폭 절감을 주 동기로** 잡는다.

---

## 2. 반드시 알아야 할 코드 사실

| 항목 | 위치 | 내용 |
|---|---|---|
| DFlash attention | `specforge/modeling/draft/dflash.py:88-140` | `q`는 draft 블록(noise) 토큰에서 나온다. **문맥 K/V(`k_ctx`, `v_ctx`)는 `target_hidden`에서 projection**된다. |
| 문맥 특징 생성 | `dflash.py:513-514` | `target_hidden = hidden_norm(fc(target_hidden))`. 계산은 forward당 1번이고, **모든 draft layer가 같은 문맥 특징을 공유**한다. 문맥 토큰끼리 서로 attention하지 않는다(문맥은 정적인 특징). |
| target 특징 추출 | `dflash.py:202-211` `extract_context_feature` | target의 여러 layer hidden을 concat한다. 예: `configs/qwen3-8b-dflash.json`은 `target_layer_ids=[1,9,17,25,33]`(36층 중), draft는 5층. |
| LittleBit 모듈 | `specforge/littlebit/{modules,packing,utils}.py` | QAT 모듈과 on-device packing. |
| LittleBit DFlash 학습 | `specforge/core/littlebit_dflash.py`, `scripts/train_littlebit_dflash{,_v2}.py` | KD loss + layer-to-layer MSE. |
| 평가 | `scripts/eval_dflash_acceptance{,_exp}.py`, `scripts/analyze_dflash_exp_results.py` | 수락 길이 τ 등. |
| 테스트 | `tests/test_utils/test_littlebit_dflash.py` | |

**핵심 제약**: 현재 DFlash draft는 프롬프트를 스스로 읽지 못한다. scorer로 쓰려면 target이 프롬프트 전체를 먼저 prefill해야 하는데, 그러면 줄이려던 비용을 이미 다 쓴 셈이다. 따라서 **target 없이 동작하는 "self 모드"가 반드시 필요하다.**

---

## 3. 제안 설계

### 3.1 한 모델, 두 모드
```
                  ┌─ self 모드:   prompt 임베딩 → (self-encoder) → 문맥 특징 ĥ
LittleBit draft ──┤
                  └─ target 모드: hidden_norm(fc(target_hidden)) → 문맥 특징 h
```
- self 모드의 문맥 특징 ĥ는 target 모드의 h와 **같은 공간**에 있어야 한다. 그래야 두 특징을 섞어 쓸 수 있다.
- self-encoder 후보:
  - (a) draft의 layer를 가중치 공유한 채 일반 causal self-attention으로 프롬프트에 돌려서 그 출력을 사용
  - (b) 얕은 전용 encoder 1~2층을 추가
  - (a)부터 시도한다.

### 3.2 Prefill 단계 (scorer)
1. self 모드로 프롬프트 전체를 처리해 ĥ를 얻는다.
2. 프롬프트 뒤에 **DFlash mask 블록(block_size개)을 붙여 forward 1회**를 돈다. 블록의 query가 미래 여러 위치의 "lookahead" 역할을 하므로, SpecPC처럼 순차 생성을 할 필요가 없다.
3. 블록 query → `k_ctx`의 attention weight를 layer와 head에 걸쳐 합산해 토큰 중요도로 쓴다. 그다음 주변 토큰끼리 묶어 평활화(chunk pooling)한다.
4. 상위 k%(그리고 마지막 window)만 target이 prefill한다. **원래 position id를 유지**한다.

### 3.3 Decode 단계 (draft)
- 섞인 문맥을 사용한다.
  - target이 남긴 토큰 → `h`(target 특징)
  - target이 버린 토큰 → `ĥ`(prefill 때 계산한 self 특징을 재사용)
- DFlash 블록 drafting → target 검증.
- 출력은 "pruning된 문맥을 가진 target" 기준으로 lossless다. 전체 문맥 대비로는 lossy이고, 이 차이는 recall로 보정한다.

### 3.4 Recall (핵심 새로움)
- 트리거 후보:
  - draft attention 중 버려진 토큰에 가는 비중이 임계값을 넘을 때
  - 수락 길이 τ가 급락할 때
- 동작: 해당 토큰(또는 그 chunk)을 target KV에 추가로 prefill해서 넣는다.

### 3.5 학습 loss (기존 DFlash/LittleBit loss에 추가)
1. **특징 distill**: `MSE/cos(ĥ, stopgrad(hidden_norm(fc(target_hidden))))`
2. **attention 중요도 distill**: target의 실제 attention 중요도(오프라인 추출)와 draft self 모드의 scoring attention 사이 KL
3. **섞인 문맥 증강**: 학습 중 `h`의 일부 위치를 무작위로 `ĥ`로 바꿔 넣는다(dropout처럼).
4. 기존 DFlash loss는 유지한다. 수락률이 떨어지면 안 된다.

---

## 4. 선행 연구 비교와 차별점

| 우리 구성요소 | 가장 가까운 선행 연구 | 남는 새로움 |
|---|---|---|
| ① draft attention을 prefill scorer로 사용 | SpecPC — *Draft-based Approximate Inference for LLMs* (2506.08373, **ICLR 2026**); SpecPrefill (2502.02789); Cross-Family SpecPrefill (2603.02631) | 거의 없음 |
| ② mask 블록으로 lookahead를 forward 한 번에 얻음 | SpecPC/SpecKV는 draft로 lookahead를 생성 | 작음(효율 개선) |
| ③ prefill 때 만든 draft 문맥을 decode에 재사용 | *Strong Drafts Need Compact Memories* (2608.30252): draft memory를 prefill 때 만들지만 target은 full KV(lossless) | 부분적으로 겹침 |
| ④ **target sparse + draft full + recall** | SparseSpec-L (2607.27735): recall 가능한 sparse KV지만 **draft가 sparse, target이 full**인 반대 방향. TriForce·MagicDec·Windowed-MTP도 모두 "target full / draft sparse" | **가장 새로움** |
| ⑤ LittleBit 극한 양자화 + NPU on-device | 찾지 못함 | 있음(단, "재탕"으로 보이지 않도록 분석으로 만들어야 함) |

**논문 중심 주장(framing)**: *"long-context에서 전체 문맥은 작은 draft가 소유하고, 비싼 target은 draft가 관리하는 sparse한 문맥만 본다."* ①②③은 이 구조를 구성하는 부품이고, 기여는 ④ + ⑤다.

**미확인 사항**: 이전 세션에서는 arxiv.org 접근이 막혀 SpecPC 원문과 2608.30252 원문을 읽지 못했다(초록과 요약 기준). **SpecPC가 같은 draft로 speculative decoding까지 하는지 반드시 원문으로 확인할 것.**

---

## 5. 실험 계획 (단계마다 계속할지 판정)

| 단계 | 내용 | 주요 지표 | 판정 기준 |
|---|---|---|---|
| **S1** | FP draft에 self 모드 + loss 1·2 추가 | target oracle 대비 recall@keep-rate, LongBench/RULER 성능 | **SpecPrefill + Qwen3-0.6B scorer**보다 같거나 나아야 계속 |
| **S2** | 섞인 문맥 decode (loss 3) | 수락 길이 τ: (a) 전체 target 문맥 (b) 남긴 토큰만 (c) 섞인 문맥 | (c)가 (a)에 근접해야 함 |
| **S3 ★** | **Recall** | pruning이 실패하기 쉬운 과제(multi-hop, 여러 needle, 코드)에서 recall 있음/없음의 정확도, recall 빈도와 비용 | **recall이 full 대비 격차를 의미 있게 메우면 탑 티어 이야기가 성립.** 안 되면 ②·⑤ 중심의 중간 규모 논문으로 축소 |
| **S4** | LittleBit 강도별 실험 (1.0 / 0.55 / 0.3 / 0.1 bpw) | scoring recall, τ, recall 트리거 정확도, perplexity를 한 그래프에 | "순위 기반 역할이 생성보다 양자화에 강하다"는 분석 figure |
| **S5** | NPU end-to-end | KV 메모리, decode tok/s, TTFT, 에너지 | 예: "8B target + 32k 문맥이 폰 메모리에 들어간다" |

**S3을 최대한 빨리 FP로 확인하는 것이 최우선이다.**

### 필수 baseline
- **SpecKV-PC + 별도 DFlash draft** (같은 메모리/NPU 예산) ← 정면으로 이겨야 하는 상대
- SpecPrefill (Qwen3-0.6B 또는 Llama-3.2-1B scorer)
- 기존 DFlash (prefill pruning 없음)
- sd.npu 계열 (on-device 비교용)

---

## 6. 리스크

| 리스크 | 대응 |
|---|---|
| 5층 draft가 날(raw) 임베딩만으로는 scorer로 약할 수 있음 | attention distill(loss 2), 그래도 부족하면 전용 encoder 1~2층 추가 |
| `ĥ`와 `h`의 분포 차이로 섞인 문맥이 불안정 | 특징 distill(loss 1) + 무작위 섞기(loss 3) |
| DFlash 수락률 저하 | 기존 DFlash loss와 함께 학습, τ를 baseline 대비로 추적 |
| self 모드 attention이 긴 프롬프트에서 길이 제곱으로 비쌈 | 층 수가 적어 감당 가능하다고 봄. 필요하면 chunk 단위 scoring |
| scoring에 attention weight가 필요해 fused kernel을 쓸 수 없음 | mask 블록 query(소수)에 대해서만 별도 계산 |
| "A+B+C 조합"이라는 리뷰 | ④ 역할 반전 + recall을 중심 주장으로, 단일 모델이어야 하는 이유(recall은 전체 문맥을 가진 draft만 가능)를 명시 |

## 7. 목표 학회 (결과 수준별)
- 역할 반전 + recall로 거의 lossless + 강한 baseline을 이김 → NeurIPS / ICLR / ICML
- 위 결과 + 실제 폰 NPU end-to-end → MobiSys / MobiCom / MLSys / ASPLOS (오히려 더 잘 맞을 수 있음)
- "합쳐도 된다 + LittleBit에서도 버틴다" 수준 → Findings / workshop

---

## 8. 다음 에이전트의 첫 작업 (제안)

1. **선행 연구 원문 확인**: SpecPC/SpecKV(2506.08373, 코드 `github.com/furiosa-ai/draft-based-approx-llm`), 2608.30252, SparseSpec(2607.27735). §4 표를 갱신하고, ④가 정말 비어 있는지 확인한다.
2. **S1 구현**:
   - `Qwen3DFlash` forward에 self 모드 경로를 추가한다. `target_hidden` 대신 self-encoder 출력을 받는다.
   - mask 블록 query → ctx attention weight를 꺼내는 scoring 함수를 만든다.
   - loss 1·2를 `specforge/core/`에 추가하고 학습 스크립트에 옵션으로 연결한다.
   - target attention 중요도를 오프라인으로 뽑는 데이터 준비 스크립트를 만든다.
   - `tests/`에 단위 테스트를 추가한다.
3. **S1 평가 스크립트**: recall@keep-rate, LongBench/RULER 일부 subset, SpecPrefill baseline.

### 참고 링크
- Speculative Prefill: https://arxiv.org/abs/2502.02789
- Cross-Family Speculative Prefill: https://arxiv.org/abs/2603.02631
- Draft-based Approximate Inference (SpecKV/SpecPC): https://arxiv.org/abs/2506.08373
- Strong Drafts Need Compact Memories: https://arxiv.org/abs/2608.30252
- SparseSpec (A Sparse Glimpse of the Whole): https://arxiv.org/abs/2607.27735
- Windowed-MTP: https://arxiv.org/abs/2607.21535
- TriForce: https://arxiv.org/abs/2404.11912
- MagicDec: https://arxiv.org/abs/2408.11049
- sd.npu: https://arxiv.org/abs/2510.15312
- llm.npu: https://arxiv.org/abs/2407.05858
- MoE-SpeQ: https://arxiv.org/abs/2511.14102 · DraftExpert: https://arxiv.org/abs/2607.24434 · Speculating Experts: https://arxiv.org/abs/2603.19289
