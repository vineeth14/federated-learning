## Abstract

Built a platform that facilitates privacy-preserving over a private, distributed data-set which facilitates Federated Learning and Differential Privacy seamlessly.
This platform integrates model training using websocket workers.

#### Framework : Pysyft

## To Run

Simply run the following commands in the terminal after cloning the repo

python3 server.py --port 8777 --id alice

python3 server.py --port 8778 --id bob

python3 server.py --port 8779 --id charlie

python3 main.py
