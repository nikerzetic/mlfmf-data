import os
import logging
import json
import io
import pstats
import cProfile
import datetime

def clean_temp_files_in_dumps(logger: logging.Logger):
    logger.info("Deleting .temp files ...")
    dir = os.path.join("dumps", "experiments")
    for file in os.listdir(dir):
        if file.endswith(".temp"):
            os.remove(os.path.join(dir, file))
            


# def read_embeddings(embeddings_file_path: str, predict_file_path: str) -> tuple[dict[str, list[float]], int]:
#     """
#     This method only works when embeddings file
#     """
#     # TODO: clean this up 
#     embeddings = {}
#     embedding_size = None
#     f = open(embeddings_file_path, "r", encoding="utf-8")
#     # Skip column names
#     f.readline()
#     for line in f:
#         parts = line.strip("\n").split("\t")
#         name = parts[0]
#         # parts[1] is the processed name, which is irrelevant here
#         embedding = parts[2:]
#         # HACK: we convert to float
#         embeddings[name] = [float(x) for x in embedding]
#         if not embedding_size:
#             embedding_size = len(embedding)
#     return embeddings, embedding_size


def read_embeddings(
    embeddings_file_path: str, predict_file_path: str, label2raw_dict_file_path: str
) -> tuple[dict[str, list[float]], int]:
    """
    ## Parameters
    - embeddings_file_path
    - predict_file_path
    - label2raw_dict_file_path

    ## Returns
    - embeddings: dictionary with labels as keys and embeddings as values (as list)
    - embeddings_size
    """

    # TODO: clean this dict stuff up

    dict_file = open(label2raw_dict_file_path, "r", encoding="utf-8")
    label2raw = json.load(dict_file)
    dict_file.close()

    embeddings = {}
    embeddings_size = None

    embeddings_file = open(embeddings_file_path, "r", encoding="utf-8")
    predict_file = open(predict_file_path, "r", encoding="utf-8")

    # Skip column names
    embeddings_file.readline()

    for embeddings_line, predict_line in zip(embeddings_file, predict_file):
        label = predict_line.split(" ")[0]
        try:
            name = label2raw[label]
        except KeyError:
            continue
        embedding = embeddings_line.strip("\n").split("\t")[1:]
        embeddings[name] = [float(x) for x in embedding]
        if not embeddings_size:
            embeddings_size = len(embedding)
    predict_file.close()
    embeddings_file.close()

    return embeddings, embeddings_size


def myprofile(func):

    def profiled_func(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        
        returned_value = func(*args, **kwargs)

        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        out_dir = "cProfile-outputs"
        os.makedirs(out_dir, exist_ok=True)
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
        out_file = os.path.join(out_dir, f"{now}.txt")
        with open(out_file, "w") as f:
            print(s.getvalue(), file=f)

        return returned_value
    
    return profiled_func