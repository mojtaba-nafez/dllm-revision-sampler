
# RCP Setup & Commands


### Intractive session  (8-hours)

```
runai submit \
  --name dllm-sampler-intractive \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash /scratch/mnafez/dllm-revision-sampler/epfl-rcp-bootstrap.sh
```

```
runai bash dllm-sampler-intractive -- bash --login
```

- Check [epfl-rcp-bootstrap.sh](epfl-rcp-bootstrap.sh) for: 
  - 8h (Intractive session)
  - Simlink for /scratch 
  - Activate conda env by default


## Submit Jobs (Train or Large Inference)

```
runai submit \
  --name dllm-eval \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate dllm && \
    cd /scratch/mnafez/dllm-revision-sampler && \
    bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Base --instruct False --limit 30
  "
```


```
runai logs dllm-eval > /scratch/mnafez/eval_output.log
```


### Runai Usefull Commands
  
```
runai logs dllm-sampler
```

```
runai bash dllm-sampler -- bash --login
```


```
runai delete job dllm-sampler
```


```
runai suspend dllm-sampler
```

```
runai resume dllm-sampler
```



```
source /scratch/mnafez/miniconda3/bin/activate
conda activate dllm
```




# Tempral Command For Run



```
runai submit \
  --name llada-instruct-baseline-sampling \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate dllm && \
    cd /scratch/mnafez/dllm-revision-sampler && \
    bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Instruct --instruct True --limit 30
  "
```



```
runai submit \
  --name llada-instruct-remasking-sampling-end \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate dllm && \
    cd /scratch/mnafez/dllm-revision-sampler && \
    bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Instruct --instruct True --limit 30
  "
```

```
runai submit \
  --name llada-instruct-gradualy-remasking-sampling \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate dllm && \
    cd /scratch/mnafez/dllm-revision-sampler && \
    bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Instruct --instruct True --limit 30
    "
```


```
runai submit \
  --name llada-instruct-gradualy-remasking-sampling2 \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate dllm && \
    cd /scratch/mnafez/dllm-revision-sampler && \
    bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Instruct --instruct True --limit 30
    "
```



```
runai submit \
  --name instruct-dynamic-remasking-no-shortcut-supp \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate dllm && \
    cd /scratch/mnafez/dllm-revision-sampler && \
    bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Instruct --instruct True --limit 30
    "
```

|Groups|Version|  Filter  |n-shot|  Metric   |   |Value |   |Stderr|
|------|------:|----------|-----:|-----------|---|-----:|---|-----:|
|bbh   |      3|get-answer|     3|exact_match|↑  |0.5395|±  |0.0159|




```
runai submit \
  --name llada-base-bhh-dynamic-remasking \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate dllm && \
    cd /scratch/mnafez/dllm-revision-sampler && \
    bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Base --instruct False --limit 30
    "
```
llada ({'pretrained': 'GSAI-ML/LLaDA-8B-Base', 'max_new_tokens': 256, 'steps': 256, 'block_size': 256, 'cfg_scale': 0.0}), gen_kwargs: ({}), limit: 30.0, num_fewshot: 5, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6333|±  |0.0895|
|     |       |strict-match    |     5|exact_match|↑  |0.6333|±  |0.0895|




```
runai submit \
  --name llada-instruc-gsm8k-dynamic-remasking \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate dllm && \
    cd /scratch/mnafez/dllm-revision-sampler && \
    bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Instruct --instruct True --limit 400
    "
```
total_number_remask tensor(0, device='cuda:0')
llada ({'pretrained': 'GSAI-ML/LLaDA-8B-Instruct', 'max_new_tokens': 512, 'steps': 512, 'block_size': 512, 'cfg_scale': 0.0, 'suppress_tokens': [], 'begin_suppress_tokens': '[126081;126348]'}), gen_kwargs: ({}), limit: 400.0, num_fewshot: 5, batch_size: 1
|  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|---------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k_cot|      3|flexible-extract|     5|exact_match|↑  |0.3750|±  |0.0242|
|         |       |strict-match    |     5|exact_match|↑  |0.2025|±  |0.0201|



```
runai submit \
  --name llada-instruc-gsm8k-baseline \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate dllm && \
    cd /scratch/mnafez/dllm-revision-sampler && \
    bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Instruct --instruct True --limit 400
    "
```
llada ({'pretrained': 'GSAI-ML/LLaDA-8B-Instruct', 'max_new_tokens': 512, 'steps': 512, 'block_size': 512, 'cfg_scale': 0.0, 'suppress_tokens': [], 'begin_suppress_tokens': '[126081;126348]'}), gen_kwargs: ({}), limit: 400.0, num_fewshot: 5, batch_size: 1
|  Tasks  |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|---------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k_cot|      3|flexible-extract|     5|exact_match|↑  |0.7825|±  |0.0207|
|         |       |strict-match    |     5|exact_match|↑  |0.5750|±  |0.0247|


## GSMK + LLaDA-Base:


```
runai submit \
  --name llada-base-bhh-dynamic-remasking \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate dllm && \
    cd /scratch/mnafez/dllm-revision-sampler && \
    bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Base --instruct False --limit 30
    "
```



```
runai submit \
  --name llada-base-gsm8k-dynamic-remasking \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate dllm && \
    cd /scratch/mnafez/dllm-revision-sampler && \
    bash examples/llada/eval.sh --model_name_or_path GSAI-ML/LLaDA-8B-Base --instruct False --limit 400
    "
```




## GSMK + LLaDA-Instruct

```
runai submit \
  --name llada-instruct-gsm8k-dynamic-remasking \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate dllm && \
    cd /scratch/mnafez/dllm-revision-sampler && \
    bash examples/llada/gsm8k.sh --model_name_or_path GSAI-ML/LLaDA-8B-Instruct --instruct True --limit 400 --sampling_strategy "dynamic-revise"
    "
```


```
runai submit \
  --name llada-instruct-gsm8k-original-sampling \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash -c "
    source /scratch/mnafez/miniconda3/etc/profile.d/conda.sh && \
    conda activate dllm && \
    cd /scratch/mnafez/dllm-revision-sampler && \
    bash examples/llada/gsm8k.sh --model_name_or_path GSAI-ML/LLaDA-8B-Instruct --instruct True --limit 400 --sampling_strategy "original-sampling"
    "
```


