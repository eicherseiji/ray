import os

from ray._private.runtime_env.protocol import ProtocolsProvider


class AnyscaleProtocolsProvider(ProtocolsProvider):
    @classmethod
    def get_protocols(cls):
        return (
            super()
            .get_protocols()
            .union(
                {
                    # Remote azure blob storage path,
                    # assumes everything packed in one zip file.
                    "azure",
                }
            )
        )

    @classmethod
    def get_remote_protocols(cls):
        return super().get_remote_protocols().union({"azure"})

    @classmethod
    def download_remote_uri(cls, protocol, source_uri, dest_file):
        if protocol == "azure":
            missing_imports = []
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError:
                missing_imports.append("azure-identity")
            try:
                from azure.storage.blob import BlobServiceClient
                from smart_open import open as open_file
            except ImportError:
                missing_imports.append("smart_open[azure]")
            if missing_imports:
                raise ImportError(
                    "You must `pip install " + " ".join(missing_imports) + "` "
                    "to fetch URIs in azure blob storage container. "
                    + cls._MISSING_DEPENDENCIES_WARNING
                )
            account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
            if not account_url:
                raise ValueError(
                    "You must provide the URL to the azure blob "
                    "storage account via env var `AZURE_STORAGE_ACCOUNT_URL` "
                    "to fetch URIs in azure blob storage container."
                )
            tp = {
                "client": BlobServiceClient(
                    account_url, credential=DefaultAzureCredential()
                )
            }
            with open_file(source_uri, "rb", transport_params=tp) as fin:
                with open_file(dest_file, "wb") as fout:
                    fout.write(fin.read())
        else:
            return super().download_remote_uri(protocol, source_uri, dest_file)
