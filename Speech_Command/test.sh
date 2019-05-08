#!/bin/bash

declare total
declare success
num_rows=10
num_columns=10

for ((i=0; i<num_rows; i++)) do
    for ((j=0; j<num_columns; j++)) do
        total[$i*10+$j]=0
        success[$i*10+$j]=0
    done
done

label=("yes" "no" "up" "down" "left" "right" "on" "off" "stop" "go")

for ((i=0; i<1; i++)) do
    for ((j=0; j<2; j++)) do
        if [ $i != $j ]
		then
			search_root="/tmp/speech_dataset"
			source_label=${label[$i]}
			search_dir="$search_root/$source_label"
			for entry in `ls $search_dir`; do
				((total[$i*10+$j]++))
				input_path="$search_dir/$entry"
				target_label=${label[$j]}
				command="python3 generate_audio_v1.py --wav_path $entry"
				ouput_path="/home/leeanghsuan/Desktop/Speech_Command/$source_label/$target_label/${success[$i,$j]}.wav"
			    commandOutput="$(python3 generate_audio_v1.py --wav_path $input_path --new_wav_path $ouput_path --target $target_label)"			   
			    if [ ${commandOutput: -1} = "s" ]
				then
					(( success[$i*10+$j] += 1))
					break
				fi
				if [ total[$i,$j] -eq 10 ]
				then
					break
				fi
			done
		fi
    done
done

for ((i=0; i<num_rows; i++))
do
    for ((j=0; j<num_columns; j++))
    do
    	if [ ${total[$i*10+$j]} -eq 0 ]
    	then
    		echo -ne "0\t"
    	else
    		accuracy=$((success[$i*10+$j] / total[$i*10+$j]))
    		echo -ne "$accuracy\t"
    	fi
    done
    echo
done