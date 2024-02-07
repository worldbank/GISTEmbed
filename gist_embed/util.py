import logging
import os
import json
import tempfile

from typing import List, Optional
from huggingface_hub import HfApi, run_as_future as run_as_future_hf
from huggingface_hub.utils._errors import HfHubHTTPError
import backoff

logger = logging.getLogger(__name__)



def background_hf_model_upload(tmp_dir, token, repo_id, commit_message):
    import shutil
    from huggingface_hub import upload_folder

    # We need to create a temporary directory that is not automatically deleted.

    with open(os.path.join(tmp_dir, "commit-info.json"), "w") as fOut:
        json.dump(dict(repo_id=repo_id, commit_message=commit_message), fOut)

    folder_url = upload_folder(
            repo_id=repo_id,
            folder_path=tmp_dir,
            commit_message=commit_message,
            token=token,
            run_as_future=False,
        )

    run_as_future_hf(shutil.rmtree, tmp_dir)

    return folder_url

# @save_to_hub_args_decorator
@backoff.on_exception(backoff.expo, [HfHubHTTPError, Exception], max_tries=6)
def save_to_hub(model,
                repo_id: str,
                organization: Optional[str] = None,
                token: Optional[str] = None,
                private: Optional[bool] = None,
                commit_message: str = "Add new SentenceTransformer model.",
                local_model_path: Optional[str] = None,
                exist_ok: bool = False,
                replace_model_card: bool = False,
                train_datasets: Optional[List[str]] = None,
                run_as_future: bool = False,
            ):
    """
    Uploads all elements of this Sentence Transformer to a new HuggingFace Hub repository.

    :param repo_id: Repository name for your model in the Hub, including the user or organization.
    :param token: An authentication token (See https://huggingface.co/settings/token)
    :param private: Set to true, for hosting a private model
    :param commit_message: Message to commit while pushing.
    :param local_model_path: Path of the model locally. If set, this file path will be uploaded. Otherwise, the current model will be uploaded
    :param exist_ok: If true, saving to an existing repository is OK. If false, saving only to a new repository is possible
    :param replace_model_card: If true, replace an existing model card in the hub with the automatically created model card
    :param train_datasets: Datasets used to train the model. If set, the datasets will be added to the model card in the Hub.
    :param organization: Deprecated. Organization in which you want to push your model or tokenizer (you must be a member of this organization).

    :return: The url of the commit of your model in the repository on the Hugging Face Hub.
    """
    if organization:
        if "/" not in repo_id:
            logger.warning(
                f"Providing an `organization` to `save_to_hub` is deprecated, please use `repo_id=\"{organization}/{repo_id}\"` instead."
            )
            repo_id = f"{organization}/{repo_id}"
        elif repo_id.split("/")[0] != organization:
            raise ValueError("Providing an `organization` to `save_to_hub` is deprecated, please only use `repo_id`.")
        else:
            logger.warning(
                f"Providing an `organization` to `save_to_hub` is deprecated, please only use `repo_id=\"{repo_id}\"` instead."
            )

    api = HfApi(token=token)
    repo_url = api.create_repo(
        repo_id=repo_id,
        private=private,
        repo_type=None,
        exist_ok=exist_ok,
    )
    if local_model_path:
        folder_url = api.upload_folder(
            repo_id=repo_id,
            folder_path=local_model_path,
            commit_message=commit_message,
            run_as_future=run_as_future,
        )
    else:
        # If run_as_future is True, we need to create a temporary directory to save the model.
        # This temporary directory should not immediately be deleted, as the upload_folder function
        # is asynchronous and will run in the background.

        # If run_as_future is False, we can use a temporary directory that is automatically deleted

        # We first implement the run_as_future=False case, as this is the most common case.

        if not run_as_future:
            with tempfile.TemporaryDirectory() as tmp_dir:
                create_model_card = replace_model_card or not os.path.exists(os.path.join(tmp_dir, 'README.md'))
                model.save(tmp_dir, model_name=repo_url.repo_id, create_model_card=create_model_card, train_datasets=train_datasets)
                folder_url = api.upload_folder(
                    repo_id=repo_id,
                    folder_path=tmp_dir,
                    commit_message=commit_message
                )
        else:
            # We need to create a temporary directory that is not automatically deleted.
            tmp_dir = tempfile.mkdtemp()
            create_model_card = replace_model_card or not os.path.exists(os.path.join(tmp_dir, 'README.md'))
            model.save(tmp_dir, model_name=repo_url.repo_id, create_model_card=create_model_card, train_datasets=train_datasets)
            folder_url = api.run_as_future(background_hf_model_upload, tmp_dir, token, repo_id, commit_message)

    refs = api.list_repo_refs(repo_id=repo_id)
    for branch in refs.branches:
        if branch.name == "main":
            return f"https://huggingface.co/{repo_id}/commit/{branch.target_commit}"
    # This isn't expected to ever be reached.
    return folder_url
