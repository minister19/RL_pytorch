# ref: https://docs.python.org/3/howto/logging.html
'''
import logging
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this, too')
logging.error('And non-ASCII stuff, too, like Øresund and Malmö')
'''

import logging


class Logger():
    def __init__(self) -> None:
        logging.basicConfig(filename='./data/rl_framework.log', filemode='w', level=logging.INFO)
