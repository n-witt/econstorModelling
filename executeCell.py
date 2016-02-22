import json
import os
import requests
import IPython
import ipykernel as kernel
connection_file_path = kernel.get_connection_file()
connection_file = os.path.basename(connection_file_path)
kernel_id = connection_file.split('-', 1)[1].split('.')[0]

def executeCell(x=0):
    ''' executes the code in cell no 'x' (zero-based indexing)
    '''
    sessions = requests.get('http://127.0.0.1:8888/api/sessions').json()
    ipynbFileName = ""
    for sess in sessions:
        if sess['kernel']['id'] == kernel_id:
            ipynbFileName = sess['notebook']['path']
            ipynbFileName = ipynbFileName.split(os.sep)[-1]
            break

    # read this notebook's file
    if ipynbFileName != "":
        with open(ipynbFileName) as f:
            nb = json.load(f)
    
    # locate cell's code
    if type(nb) == dict:
        try:
            code = ""
            if nb[u'cells'][x][u'cell_type'] == u'code':
                for s in nb[u'cells'][x]['source']:
                    code += s
            else:
                raise TypeError("The cell you request is not of type 'code'")
        except IndexError:
            raise IndexError('No cell #' + str(x))
    # execute
    get_ipython().run_cell(code)
