#!/bin/bash

# Usage: bash submit_pipeline.sh <number>
# e.g.:  bash submit_pipeline.sh 10

INIT_ID=$(sbatch --parsable initialize.sh $1)
echo "Submitted initialize.sh as job $INIT_ID"

ARRAY_ID=$(sbatch --parsable --dependency=afterok:$INIT_ID array.sh $1)
echo "Submitted array.sh as job $ARRAY_ID"

ACCUM_ID=$(sbatch --parsable --dependency=afterany:$ARRAY_ID accumulate.sh $1)
echo "Submitted accumulate.sh as job $ACCUM_ID"

#FINAL_ID=$(sbatch --parsable --dependency=afterok:$ACCUM_ID final.sh $1)
#echo "Submitted final.sh as job $FINAL_ID"

echo ""
echo "Pipeline submitted:"
echo "  $INIT_ID (initialize) --> $ARRAY_ID (array) --> $ACCUM_ID (accumulate)" #--> $FINAL_ID (final)