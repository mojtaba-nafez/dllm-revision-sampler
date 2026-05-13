
# RCP Setup & Commands


### Intractive session  (8-hours)

```
runai submit \
  --name dllm-sampler \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --gpu 1 \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command -- bash /scratch/mnafez/dllm-revision-sampler/epfl-rcp-bootstrap.sh
```

- Check [epfl-rcp-bootstrap.sh](epfl-rcp-bootstrap.sh) for: 
  - 8h (Intractive session)
  - Simlink for /scratch 
  - Activate conda env by default

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