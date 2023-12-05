# docker run -v ${REPO_PATH}:/workspace/pointnetplusplus -v ${DATASETS_PATH}:/workspace/datasets --runtime nvidia -it pytorch3d

FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

RUN apt update; apt install dialog apt-utils; apt install less

RUN conda install -c fvcore -c iopath -c conda-forge -c pytorch3d fvcore iopath pytorch3d wandb
