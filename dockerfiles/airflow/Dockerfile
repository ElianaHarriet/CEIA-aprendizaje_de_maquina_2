FROM apache/airflow:3.0.2
COPY requirements.txt /requirements.txt
COPY bootstrap.sh /opt/airflow/bootstrap.sh

USER root
RUN apt-get update --allow-insecure-repositories
RUN apt-get install --allow-unauthenticated -y git
RUN apt-get autoremove -yqq --purge
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install --no-cache-dir apache-airflow[virtualenv]==3.0.2 -r /requirements.txt

ENTRYPOINT ["/bin/bash", "/opt/airflow/bootstrap.sh"]
CMD ["bash"]
