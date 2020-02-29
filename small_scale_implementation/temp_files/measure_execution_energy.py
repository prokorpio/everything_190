# Notes
# from "nvidia-smi --help-query-gpu" it is said that power.draw query is
# accurate only to +/- 5 watts.

# functionality: 
# call as class in py file
# run gpu tracker in background
# end run in pyfile

# expansion:
# plot realtime in tensorboard

import subprocess
import time
import signal
import os

class TrackGPUPower():
    ''' Record GPU power in real time using 'nvidia-smi' 
        Adapted from ECC: hyang1990/energy_constrained_compression
    '''
    def __init__(self, gpuid=0, rm_watts_offset=False, file='power.txt',
                       time_stamp=False, loop_time=1, grain='sec'):
        self.filename = file
        # handle idle watts
        self.idle_watts = 0.0
        if not rm_watts_offset:
            cmd = 'nvidia-smi --query-gpu=power.draw ' + \
                    '--id={} '.format(gpuid) + \
                    '--format=csv,noheader'
            for _ in range(10):
                self.idle_watts += float(subprocess.getoutput(cmd)[:-1]) 
                                                    # remove last letter W
            self.idle_watts /= 10 # get ave of 10 samples

        # set command
        self.cmd = ['nvidia-smi'] 

        if time_stamp: 
              self.cmd += ['--query-gpu=power.draw,timestamp'] 
        else:
              self.cmd += ['--query-gpu=power.draw']

        if grain == 'sec':
            self.cmd += ['--loop={}'.format(loop_time)] 
        elif grain == 'ms':
            self.cmd += ['--loop-ms={}'.format(loop_time)] 

        # the rest
            self.cmd += ['--id={}'.format(gpuid)] + \
                        ['--filename={}'.format(file)] + \
                        ['--format=csv,noheader']

    def start(self):
        self.proc = subprocess.Popen(self.cmd) 

    def end(self):
        self.proc.send_signal(signal.SIGINT)
        self.proc.wait() # block for termination

        # check if file exists w/ content
        assert os.stat(self.filename).st_size > 0
        # note, may be fooled if there's an older version of file

        return self.idle_watts

if __name__ == '__main__':
    # test
    power_tracker = TrackGPUPower()
    power_tracker.start()
    for i in range(5):
        time.sleep(1)
        print('sleeping',i)
    print(power_tracker.end())
