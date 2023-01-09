from lazydocs import generate_docs
import shutil


docs = './docs'
shutil.rmtree(docs)
generate_docs([
    "mnn.layer"
    ], output_path=docs, watermark=False)
