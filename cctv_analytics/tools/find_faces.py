import sys
import json
import logging
from pymilvus import MilvusClient, DataType
from apis.deepface_setup import DeepfaceAPI, encode_image
from utils.config import genconf, configure_logging

configure_logging()
config = genconf()


def run():
    # Check if an argument is provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        logging.error("Image path argument is required.")

    deepface_conn = DeepfaceAPI(config.deepface.url)
    client_milvus = MilvusClient(uri=config.milvus.uri, token=config.milvus.token)

    img = encode_image(image_path)
    embeddings = deepface_conn.represent(
        img,
        config.deepface.deepface_model_name,
        config.deepface.detector_name,
    )

    for embedding in embeddings:
        # Single vector search
        res = client_milvus.search(
            collection_name="master_faces",  # Replace with the actual name of your collection
            anns_field="image_embeddings",
            # Replace with your query vector
            data=[embedding["embedding"]],
            limit=15,  # Max. number of search results to return
            search_params={"metric_type": "IP", "params": {}},  # Search parameters
        )

        # Convert the output to a formatted JSON string
        result = json.dumps(res, indent=4)
        logging.info(result)
