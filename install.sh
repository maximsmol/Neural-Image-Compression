#!/bin/bash

set -e

pip3 install virtualenv
test -d "venv" || (virtualenv venv && source venv/bin/activate && pip3 install -r requirements.txt && deactivate)
