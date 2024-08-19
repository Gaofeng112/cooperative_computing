#!/bin/bash
./flatc -t --strict-json --defaults-json -o workdir ./schema.fbs -- net/converted_diffusion_model.tflite
python3 extract_json_lib.py
for i in {0..96}
do
  ./flatc -o workdir -b ./schema.fbs workdir/subgraphs/cpusubgraph"$i".json
  ./flatc -o workdir -b ./schema.fbs workdir/subgraphs/npusubgraph"$i".json
done
for i in {97..123}
do
  ./flatc -o workdir -b ./schema.fbs workdir/subgraphs/npusubgraph"$i".json
done
