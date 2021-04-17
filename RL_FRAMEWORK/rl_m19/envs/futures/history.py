# coding=utf-8
from datetime import datetime, timedelta
import os
import json


class History:
    @staticmethod
    def fetch_history(symb, freq, start_time, end_time):
        pass


if __name__ == '__main__':
    symb = 'SHFE.RB'
    freq = '5mins'
    start_time = datetime.now() + timedelta(days=-7)
    end_time = datetime.now() + timedelta(days=-1)
    History.fetch_history(symb, freq, start_time, end_time)
