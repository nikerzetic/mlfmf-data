import os
import logging

def clean_temp_files_in_dumps(logger: logging.Logger):
    logger.info("Deleting .temp files ...")
    dir = os.path.join("dumps", "experiments")
    for file in os.listdir(dir):
        if file.endswith(".temp"):
            os.remove(os.path.join(dir, file))
            


def read_embeddings(embeddings_file_path: str) -> tuple[dict[str, list[float]], int]:
    embeddings = {}
    embedding_size = None
    f = open(embeddings_file_path, "r", encoding="utf-8")
    # Skip column names
    f.readline()
    for line in f:
        parts = line.strip("\n").split("\t")
        name = parts[0]
        # parts[1] is the processed name, which is irrelevant here
        embedding = parts[2:]
        # HACK: we convert to float
        embeddings[name] = [float(x) for x in embedding]
        if not embedding_size:
            embedding_size = len(embedding)
    return embeddings, embedding_size


# def read_embeddings(
#     library_name: str, save_to_file: str = None
# ) -> tuple[dict[str, list[float]], int]:
#     """
#     ## Parameters
#     - library_name
#     - save_to_file: path to file for json dump

#     ## Returns
#     - embeddings: dictionary with labels as keys and embeddings as values (as list)
#     - embeddings_size
#     """
#     print(f"Reading embeddings for {library_name}...")
#     embeddings_file_path = os.path.join(
#         "data", "embeddings", "code2seq", f"{library_name}.tsv"
#     )
#     predict_file_path = os.path.join("data", "code2seq", library_name, "predict.c2s")
#     embeddings = {}
#     embeddings_size = None

#     embeddings_file = open(embeddings_file_path, "r", encoding="utf-8")
#     predict_file = open(predict_file_path, "r", encoding="utf-8")
#     embeddings_file.readline()
#     for embeddings_line, predict_line in zip(embeddings_file, predict_file):
#         label = predict_line.split(" ")[0]
#         embedding = embeddings_line.strip("\n").split("\t")[1:]
#         embeddings[label] = embedding
#         if not embeddings_size:
#             embeddings_size = len(embedding)
#     predict_file.close()
#     embeddings_file.close()

#     if save_to_file:
#         to_print = ["\t".join([f"\n{name}"] + e) for name, e in embeddings.items()]
#         components = "\t".join([f"x{i}" for i in range(embeddings_size)])
#         with open(save_to_file, "w", encoding="utf-8") as f:

#             f.write(f"name\tprocessed\tembedding\t{components}")
#             f.writelines(to_print)

#     return embeddings, embeddings_size
