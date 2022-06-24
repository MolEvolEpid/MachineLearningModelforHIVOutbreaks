#!/bin/bash
#KEY="Exponential_exits_synth"
#KEY="Uniform_exits_synth"
#KEY="Uniform_exits_mu67_synth"
KEY="Exponential_exits_mu67_synth"
CORE="*Example_Data/${KEY}"

for SIZE in "15" "20" "30" "40" "50" 
do
    TEST="${CORE}*data_${SIZE}*TEST"
    TRAIN="${CORE}*data_${SIZE}*TRAIN"
    echo $TEST
    echo "${TEST}*W${SIZE}-*.mat"
    echo $(find -wholename "${TEST}*W${SIZE}*.mat")    
    python3 ./MakeAndTrain_CLI.py --test $(find '../' -wholename "${TEST}*W${SIZE}*.mat")  --train $(find '../' -wholename "${TRAIN}*W${SIZE}*.mat") --savekey $KEY --ordering "None"
    echo "Python3 called and run"
done
source deactivate
echo "Conda Env deactivated"

