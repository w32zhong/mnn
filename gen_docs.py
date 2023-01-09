from lazydocs import generate_docs
import shutil
import os
import mnn.layer as layers


docs = './docs'
if os.path.exists(docs):
    shutil.rmtree(docs)

modules = []
for layer in dir(layers):
    if layer.startswith('__'):
        continue
    elif layer in ['Tensor', 'BaseLayer']:
        continue
    modules.append(f'mnn.layer.{layer}')

modules.append('mnn.seq_layers')

generate_docs(modules, output_path=docs,
    watermark=False, remove_package_prefix=True)
