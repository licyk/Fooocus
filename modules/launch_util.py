import logging
import os
import shutil

logging.getLogger("torch.distributed.nn").setLevel(logging.ERROR)
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())


def delete_folder_content(folder, prefix=None):
    result = True

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'{prefix}Failed to delete {file_path}. Reason: {e}')
            result = False

    return result
