#!/bin/bash

# @sky copy the code in Librispeech Tutorial before computing fmlrr features
#      and modify it for TIMIT dataset

# if [ "$#" -ne 2 ]; then
#   echo "Usage: $0 <chunk> <gmmdir>"
#   echo "e.g.: $0 train(/test-set/in_label) exp/TIMIT_MLP_basic_forward"
#   exit 1
# fi

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
. ./path.sh ## Source the tools/utils (import the queue.pl)

chunk=train
#chunk=dev # Uncomment to process dev
#chunk=test # Uncomment to process test
#chunk=test_set/in_label
#chunk=test_1

gmmdir=exp/tri4b

dir=fmllr/$chunk
# @sky TIMIT dataset has already make the feast. Don't need to make the feature separately
steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
    --transform-dir $gmmdir/decode_tgsmall_$chunk \
        $dir data/$chunk $gmmdir $dir/log $dir/data || exit 1
        
compute-cmvn-stats --spk2utt=ark:data/$chunk/spk2utt scp:fmllr/$chunk/feats.scp ark:$dir/data/cmvn_speaker.ark
