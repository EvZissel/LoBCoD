#!/bin/sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/
matlab $* -singleCompThread -r "addpath('./build/'); addpath('./test_release'); setenv('MKL_NUM_THREADS','4'); setenv('MKL_SERIAL','NO');setenv('MKL_DYNAMIC','FALSE');"
