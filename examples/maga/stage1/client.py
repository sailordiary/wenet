import io
import os

import boto3
from botocore.exceptions import EndpointConnectionError, NoCredentialsError
from retry import retry


class FileHandler:

    def __init__(self,
                 aws_access_key=None,
                 aws_secret_key=None,
                 endpoint=None,
                 retry_times=5):
        self.s3_client = None
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.endpoint_url = endpoint
        self.retry_times = retry_times
        if aws_access_key and aws_secret_key and endpoint:
            session = boto3.Session()
            self.s3_client = session.client(
                service_name='s3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                endpoint_url=endpoint,
            )
        self.file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def open(self, path):
        file_bytes = self.get_file_bytes(path)
        self.file = io.BytesIO(file_bytes)
        return self.file

    def get_file_bytes(self, path):
        if path.startswith('ceph://') or path.startswith('s3://'):
            return self._download_from_ceph_s3(path)
        else:
            return self._read_local_file(path)

    def _read_local_file(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as file:
                return file.read()
        else:
            raise FileNotFoundError(f"The local file {path} does not exist.")

    def _retry(self, func):

        @retry(tries=self.retry_times, delay=2, backoff=2)
        def wrapped(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapped

    def _download_from_ceph_s3(self, path):
        return self._retry(self._download_from_ceph_s3_impl)(path)

    def _download_from_ceph_s3_impl(self, path):
        if self.s3_client is None:
            raise ValueError("S3 client is not configured properly.")

        bucket, key = self._parse_ceph_s3_path(path)

        try:
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            data = response['Body'].read()
            response['Body'].close()
            return data
        except NoCredentialsError as e:
            raise NoCredentialsError(
                "Credentials for S3 are not available.") from e
        except EndpointConnectionError as e:
            raise EndpointConnectionError(
                f"Failed to connect to endpoint: {self.endpoint_url}") from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to download file from {path}. Error: {str(e)}") from e

    def _parse_ceph_s3_path(self, path):
        if path.startswith('ceph://'):
            path = path[len('ceph://'):]
        elif path.startswith('s3://'):
            path = path[len('s3://'):]

        bucket, key = path.split('/', 1)
        return bucket, key

    def get_state(self):
        return {
            'aws_access_key': self.aws_access_key,
            'aws_secret_key': self.aws_secret_key,
            'endpoint_url': self.endpoint_url,
            'retry_times': self.retry_times,
        }

    def set_state(self, state):
        self.aws_access_key = state.get('aws_access_key')
        self.aws_secret_key = state.get('aws_secret_key')
        self.retry_times = state.get('retry_times', 5)
        self.endpoint_url = state.get('endpoint_url')
        if (self.aws_access_key and self.aws_secret_key and self.endpoint_url):
            session = boto3.Session()
            self.s3_client = session.client(
                service_name='s3',
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
                endpoint_url=self.endpoint_url,
            )
        else:
            self.s3_client = None
