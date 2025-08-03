from database import retrieve
from app.utils import compute_dense_vector
from app.utils.helper import helper

if __name__ == "__main__":
    COLLECTION_NAME = "NcGPT"
    
    query = compute_dense_vector("What is my name?")
    
    points = retrieve(collection_name=COLLECTION_NAME, query=query)
    
    result = helper.extract_points(points=points)
    
    print(result)