FROM public.ecr.aws/lambda/python:3.12

# Copy root, including .env
COPY . ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt

# Copy function code
COPY functions/get-recommendations/handler.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "handler.handler" ]