import sys
sys.path.append('../')

import os
from logparser import Drain

def parse(logname,format_index):
    input_dir = os.path.join("../data/", logname)
    output_dir = os.path.join("../data_preprocessed/", logname)
    formats=['<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>','<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>']
    log_format=formats[format_index]
    # log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'
    # log_format = '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>'
    regex = [
        r'(0x)[0-9a-fA-F]+', #hexadecimal
        r'\d+.\d+.\d+.\d+',
        r'/\w+( )$'
        r'\d+'
    ]

    logparser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, rex=regex)
    logparser.parse(logname+'.log')


if __name__ == "__main__":
    logname = "BGL"
    logfile = "BGL.log"
    parse(logname,logfile)

