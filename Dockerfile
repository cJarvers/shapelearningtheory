FROM nvcr.io/nvidia/pytorch:23.07-py3

RUN pip install torchmetrics pytorch-lightning
RUN pip install rsatoolbox pandas seaborn