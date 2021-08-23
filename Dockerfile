# kaggleのpython環境をベースにする
FROM gcr.io/kaggle-images/python:v102

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# ライブラリの追加インストール
RUN pip install -U pip && \
    pip install fastprogress japanize-matplotlib

RUN conda install -y \
  nodejs

#tqdm
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension \
 && jupyter labextension install @jupyter-widgets/jupyterlab-manager