# EPFL RCP / RunAI Workflow (Course EE-628)

## 0. SSH Login

```bash
ssh nafez@jumphost.rcp.epfl.ch
```

Check your user info:

```bash
id
```

Example output:

```bash
uid=309139(nafez) gid=30299(...)
```

---

## 1. Docker Image Tag Format

All images must follow this format:

```bash
registry.rcp.epfl.ch/<project>/<container_name>:<version>
```

---

## 2. Prepare Project

```bash
touch requirements.txt
```

---

## 3. Build Docker Image

```bash
sudo docker build -t registry.rcp.epfl.ch/rcp-runai-course-ee-628_appgrpu-nafez/my-toolbox:v0.2 \
  -t registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2 \
  --build-arg LDAP_GROUPNAME=rcp-runai-course-ee-628_AppGrpU \
  --build-arg LDAP_GID=85219 \
  --build-arg LDAP_USERNAME=nafez \
  --build-arg LDAP_UID=309139 \
  .
```

---

## 4. Push Docker Image

```bash
sudo docker push registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.2
```

---

## 5. Run Docker Locally

```bash
sudo docker run -it registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.1 sh
```

---

## 6. Docker Logout (optional)

```bash
sudo docker logout registry.rcp.epfl.ch
```

---

## 7. Jumphost & Cluster Access

```bash
ssh nafez@jumphost.rcp.epfl.ch
```

---

## 8. Shared Scratch Directory

```bash
/mnt/course-ee-628/scratch/
```

Example contents:

```bash
hf_home  home  homes  leixu  naimer  outputs  post_train  pre_train  runs  shared  users
```

---

## 9. RunAI Job Submission

```bash
runai submit \
  --name my-demo-job \
  --image registry.rcp.epfl.ch/dllm-sampling/my-toolbox:v0.1 \
  --gpu 0.1 \
  --environment MY_ENV_VAR="A test ENV variable" \
  --existing-pvc claimname=course-ee-628-scratch,path=/scratch \
  --existing-pvc claimname=home,path=/home/mnafez \
  --command \
  -- /