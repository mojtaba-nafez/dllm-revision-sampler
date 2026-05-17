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

Dockerfile

```
#####################################
# RCP CaaS requirement (Image)
#####################################
# The best practice is to use an image
# with GPU support pre-built by Nvidia.
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/

# For example, if you want to use an image with pytorch already installed
FROM nvcr.io/nvidia/pytorch:25.12-py3

ARG DEBIAN_FRONTEND=noninteractive

# TUTORIAL ONLY
# In this example we'll use a smaller image to speed up the build process.
# Basic image based on ubuntu 22.04
# FROM docker.io/library/ubuntu:22.04

#####################################
# RCP CaaS requirement (Storage)
#####################################
# Create your user inside the container.
# This block is needed to correctly map
# your EPFL user id inside the container.
# Without this mapping, you are not able
# to access files from the external storage.
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}
#####################################

# Copy your code inside the container
RUN mkdir -p /home/${LDAP_USERNAME}
COPY ./ /home/${LDAP_USERNAME}

# Set your user as owner of the new copied files
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

# Install required packages
RUN apt update
RUN apt install python3-pip -y

RUN apt update && apt install -y \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    git-lfs \
    curl \
    ca-certificates \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    vim \
    htop \
    tmux \
    unzip \
    zip \
 && rm -rf /var/lib/apt/lists/*
 

# Set the working directory in your user's home
WORKDIR /home/${LDAP_USERNAME}
USER ${LDAP_USERNAME}

RUN pip install --upgrade pip setuptools wheel

# Install additional dependencies
RUN pip install -r requirements.txt
# OR/AND
# optional extra packages
RUN pip install \
    matplotlib \
    numpy \
    scipy \
    torch \
    torchvision \
    torchaudio
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