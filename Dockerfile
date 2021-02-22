# To enable ssh & remote debugging on app service change the base image to the one below
# FROM mcr.microsoft.com/azure-functions/python:3.0-python3.8-appservice
FROM mcr.microsoft.com/azure-functions/python:3.0-python3.8

ENV AzureWebJobsScriptRoot=/home/site/wwwroot \
    AzureFunctionsJobHost__Logging__Console__IsEnabled=true

ENV CURRENT_MODEL catordog_model_07_.h5

COPY requirements.txt /
RUN python -m pip install --upgrade pip
RUN pip install -r /requirements.txt

COPY . /home/site/wwwroot