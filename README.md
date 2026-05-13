
# RCP Commands

### Docker Setup


```
ln -s /scratch/mnafez/ ~/scratch
```


### Intractive session  (28,800 ~ 8-hours)

```
runai submit \
  --name dllm-sampler \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --gpu 1 \
  --environment MY_ENV_VAR="A test ENV variable" \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command \
  -- /bin/bash -ic "sleep 28800"
```


```
runai bash dllm-sampler
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