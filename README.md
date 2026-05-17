
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